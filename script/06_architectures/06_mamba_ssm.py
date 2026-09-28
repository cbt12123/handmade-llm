"""
06 · 非注意力路线：状态空间模型（SSM）与 Mamba
================================================
注意力的复杂度是 O(T²)，SSM 想把序列建模拉回 O(T)：

  连续系统：  h'(t) = A·h(t) + B·x(t)      y(t) = C·h(t)
  离散化后：  h_t = Ā·h_{t−1} + B̄·x_t      y_t = C·h_t

  如果 Ā、B̄、C 与输入无关（LTI 系统），整个序列可以写成【一次卷积】：
      y = x ⊛ K,   K = (C·B̄, C·Ā·B̄, C·Ā²·B̄, ...)     ← 训练时超快，用 FFT 即可

Mamba 的关键改动：让 B、C、Δ 依赖输入（selective scan）
  → 不再是卷积，不能 FFT，但可以用【并行扫描 associative scan】在 O(log T) 深度内算完
  → 推理时是 O(1) 状态更新，天然支持无限长上下文

本脚本：
  1. 手写 SSM 的递归形式与卷积形式，验证二者等价
  2. 实现 selective scan（Mamba 的核心），对比与注意力的复杂度
  3. 揭示"线性注意力"与 SSM 的联系
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import check_close, require_torch, section, set_seed, subsection, timer

set_seed(13)
torch = require_torch("06_mamba_ssm")
nn = torch.nn
F = torch.nn.functional


def discretize(A, B, delta):
    """零阶保持（ZOH）离散化：Ā = exp(Δ·A), B̄ = (Ā − I)/A · B ≈ Δ·B（A 为对角时）"""
    A_bar = torch.exp(delta * A)                       # (d_state,)
    B_bar = delta * B                                   # 一阶近似（Mamba 实际用的就是 Δ·B）
    return A_bar, B_bar


def ssm_recurrent(x, A, B, C, delta):
    """递归形式：O(T·N·d)，逐步更新状态。x: (T,) 单通道示例"""
    T = x.shape[0]
    h = torch.zeros_like(A)
    ys = []
    for t in range(T):
        A_bar, B_bar = discretize(A, B, delta[t])
        h = A_bar * h + B_bar * x[t]
        ys.append((C * h).sum())
    return torch.stack(ys)


def ssm_convolution(x, A, B, C, delta):
    """卷积形式：先算出核 K，再一次性卷积（假设 Δ 不随 t 变化）。"""
    T = x.shape[0]
    A_bar, B_bar = discretize(A, B, delta[0])
    K = torch.stack([(C * (A_bar ** k) * B_bar).sum() for k in range(T)])
    return F.conv1d(x.view(1, 1, T), K.view(1, 1, T).flip(-1), padding=T - 1)[0, 0, :T]


section("1) 递归形式 == 卷积形式（LTI 系统的美妙性质）")
N_STATE = 8
T_SEQ = 24
torch.manual_seed(0)
A = -torch.rand(N_STATE) - 0.5                 # 必须是负的（系统稳定）
B = torch.randn(N_STATE)
C = torch.randn(N_STATE)
x = torch.randn(T_SEQ)
delta = torch.full((T_SEQ,), 0.3)              # 固定的采样步长
y_rec = ssm_recurrent(x, A, B, C, delta)
y_conv = ssm_convolution(x, A, B, C, delta)
check_close(y_rec, y_conv, tol=1e-4, name="递归 SSM vs 卷积 SSM")
print("""
  训练时用卷积形式（可并行、快），推理时用递归形式（O(1) 状态、省显存）
  —— 这个"训练/推理两种等价实现"的技巧叫【dual form】，是 S4/Mamba 的精髓。
""")

section("2) Mamba 的选择性扫描（selective scan）")
def selective_scan(x, A, B_fn, C_fn, delta_fn):
    """Δ、B、C 都依赖输入 → 无法卷积，只能扫描。这里给出串行版本（便于理解）。"""
    T = x.shape[0]
    h = torch.zeros_like(A)
    ys = []
    for t in range(T):
        xt = x[t]
        delta_t = F.softplus(delta_fn(xt))          # Δ 必须为正
        Bt, Ct = B_fn(xt), C_fn(xt)
        A_bar = torch.exp(delta_t * A)
        B_bar = delta_t * Bt
        h = A_bar * h + B_bar * xt
        ys.append((Ct * h).sum())
    return torch.stack(ys)


torch.manual_seed(1)
d_model, d_state = 16, 8
in_proj = nn.Linear(1, 3 * d_state + 1)          # 生成 Δ, B, C（简化：都从标量 x_t 生成）
A = -torch.rand(d_state) - 0.5


def run_selective(x):
    params = in_proj(x.view(-1, 1))               # (T, 3·d_state + 1)
    delta = params[:, 0]
    B_all = params[:, 1:1 + d_state]
    C_all = params[:, 1 + d_state:1 + 2 * d_state]
    return selective_scan(x, A,
                          B_fn=lambda xt: in_proj(xt.view(1, 1))[0, 1:1 + d_state],
                          C_fn=lambda xt: in_proj(xt.view(1, 1))[0, 1 + d_state:1 + 2 * d_state],
                          delta_fn=lambda xt: in_proj(xt.view(1, 1))[0, 0])


y_sel = run_selective(x)
print(f"  选择性扫描输出范数 = {y_sel.norm():.4f}（输入长度 T={T_SEQ}）")
print("""
  由于每一步的 Δ/B/C 都不同，不能再写成卷积 → Mamba 用【并行扫描】解决：
    把 (a, b) 定义成"仿射变换" h ← a·h + b，定义结合律运算 ⊗：
        (a₁,b₁) ⊗ (a₂,b₂) = (a₂a₁, a₂b₁ + b₂)
    这个运算满足结合律 → 可以像前缀和一样并行扫描，深度 O(log T)
  这就是 Blelloch scan，也是 Mamba 能在 GPU 上跑得比 LSTM 快几十倍的原因。
""")

section("3) 复杂度对比：注意力 vs SSM")
print(f"{'序列长度 T':<12}{'注意力 FLOPs':>18}{'SSM(Mamba) FLOPs':>20}{'注意力/SSM':>12}")
d = 64
for T in [1024, 4096, 16384, 65536, 262144]:
    attn = 2 * T * T * d                     # QKᵀ + AV
    ssm = 2 * T * d * 16                     # 状态维度 16 的扫描
    print(f"  {T:<10}{attn / 1e9:>16.2f}G{ssm / 1e9:>18.3f}G{attn / ssm:>12.1f}")
print("""
  T 很大时注意力彻底不可行；SSM 是线性的 —— 这就是"无限上下文"的理论基础。
  但 SSM 的常数（状态维度 N）会限制它的"记忆容量"：
    状态大小 = d_model × N × 层数，是一个【固定容量】的压缩记忆
    注意力则保留全部历史（O(T) 增长）→ 精确检索任务上注意力更强
  实测结论（2024~2025）：纯 SSM 在长文检索上仍弱于注意力，
    所以主流方案是【混合架构】（Jamba / Samba / Mamba-2-hybrid）：
    大部分层用 SSM 处理局部、少数层用注意力做精确检索。
""")

section("4) 线性注意力：SSM 与注意力的桥梁")
print("""
  去掉 softmax 的注意力：
      Attention = (Q Kᵀ) V   →   Q (Kᵀ V)
  左边要先算 T×T 的矩阵；右边先算 KᵀV（d×d，与 T 无关）→ 复杂度变成 O(T·d²)
  这正是"线性注意力"（Linear Transformer / Performer / RWKV / RetNet）的思路。
  而 Mamba-2 证明：线性注意力与 SSM 在数学上是【对偶】的（SSD：Structured State-space Duality）
  → 可以用同一套矩阵乘法（matmul）单元实现，直接吃满 TensorCore。
""")
# 数值验证：(QKᵀ)V == Q(KᵀV)
torch.manual_seed(2)
T2, d2 = 32, 16
Q, K, V = torch.randn(T2, d2), torch.randn(T2, d2), torch.randn(T2, d2)
lhs = (Q @ K.T) @ V
rhs = Q @ (K.T @ V)
check_close(lhs, rhs, tol=1e-4, name="(QKᵀ)V vs Q(KᵀV)（无 softmax 时等价）")

section("5) 一句话总结")
print("""
  注意力 = 无限记忆、O(T²) 代价
  SSM    = 固定容量记忆、O(T) 代价、O(1) 推理状态
  未来大概率属于"混合"：用 SSM 扛长度，用注意力扛精度。
  对工程师的直接建议：除非你在做长序列（>100k）研究，
  否则现阶段优先用成熟的注意力生态（FlashAttention + KV Cache 已经很快了）。
""")
