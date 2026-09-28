"""
06 · FlashAttention：不落地 T×T 矩阵的注意力
==============================================
朴素注意力的显存问题：
  必须显式算出 S = QKᵀ (T×T) 和 P = softmax(S) (T×T) 才能算 P·V
  → T=8192 时，单头 fp16 就是 8192²×2 = 128MB，32 头 × batch 直接爆显存

FlashAttention 的两个核心技术：
  ① 分块（tiling）：把 Q/K/V 切成小块，一次只算一块，绝不写回全局显存
  ② 在线 softmax（online softmax）：流式地维护 running max 与 running sum，
     保证"边算边归一化"，最终结果与一次性 softmax 完全等价

本脚本用 PyTorch 写出朴素版与 Flash（在线 softmax）版，验证数值等价 + 显存对比。
（真实 CUDA 实现在第 7 章：还要做 shared memory 分块、寄存器优化、重计算等）
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import bytes_str, check_close, require_torch, section, set_seed, subsection, timer

set_seed(23)
torch = require_torch("06_flash_attention")
F = torch.nn.functional


def attention_naive(Q, K, V, is_causal=True):
    """标准实现：完整落地 T×T 的分数矩阵。"""
    d = Q.shape[-1]
    S = Q @ K.transpose(-2, -1) / (d ** 0.5)              # (H, T, T) ← 显存杀手
    if is_causal:
        T = S.shape[-1]
        mask = torch.ones(T, T, dtype=torch.bool).tril()
        S = S.masked_fill(~mask, float("-inf"))
    P = torch.softmax(S, dim=-1)                          # 又一个 (H, T, T)
    return P @ V, S, P


def attention_flash(Q, K, V, block_size=64, is_causal=True):
    """在线 softmax 的分块注意力（算法层面等价于 FlashAttention）。"""
    T, d = Q.shape[0], Q.shape[-1]
    O = torch.zeros_like(Q)
    m = torch.full((T, 1), -float("inf"))                 # running max
    l = torch.zeros((T, 1))                               # running sum(exp)

    for j in range(0, T, block_size):                     # 遍历 K/V 块
        j_end = min(j + block_size, T)
        Kj, Vj = K[j:j_end], V[j:j_end]
        Sij = Q @ Kj.T / (d ** 0.5)                       # (T, Bj) —— 只有块大小
        if is_causal:
            rows = torch.arange(T)[:, None]
            cols = torch.arange(j, j_end)[None, :]
            Sij = Sij.masked_fill(cols > rows, float("-inf"))
        m_new = torch.maximum(m, Sij.max(dim=-1, keepdim=True).values)
        # 修正因子：旧的累加量要乘 exp(m_old − m_new)
        correction = torch.exp(m - m_new)
        Pij = torch.exp(Sij - m_new)
        l = correction * l + Pij.sum(dim=-1, keepdim=True)
        O = correction * O + Pij @ Vj
        m = m_new
    return O / l, m, l


section("1) 正确性：朴素 vs 在线 softmax")
T, D = 256, 32
torch.manual_seed(0)
Q = torch.randn(T, D)
K = torch.randn(T, D)
V = torch.randn(T, D)

O_naive, S, P = attention_naive(Q, K, V)
O_flash, _, _ = attention_flash(Q, K, V, block_size=32)
check_close(O_naive, O_flash, tol=1e-5, name="朴素注意力 vs Flash（在线 softmax）")
print("  完全相同 → FlashAttention 是【精确】算法，不是近似（这是它能被广泛采用的根本原因）")

section("2) 显存对比：T×T 矩阵到底多大")
print(f"{'序列长度 T':<12}{'朴素注意力 S+P (fp16)':>24}{'Flash (仅 O)':>18}{'节省倍数':>12}")
for t in [1024, 2048, 8192, 32768, 131072]:
    naive = 2 * t * t * 2                                  # S 和 P 两份，fp16
    flash = t * 32 * 2                                     # 只存输出（d=32）
    print(f"  {t:<10}{bytes_str(naive):>24}{bytes_str(flash):>18}{naive / flash:>12.0f}")
print("""
  注意：上表只算了单头。真实模型是 (层数 × 头数 × batch) 倍 ——
  所以"8B 模型、32k 上下文"用朴素注意力根本跑不起来。
""")

section("3) 实测：三种块大小下的误差与耗时")
with timer("朴素注意力 (T=2048)"):
    Q2, K2, V2 = torch.randn(2048, 64), torch.randn(2048, 64), torch.randn(2048, 64)
    o1, _, _ = attention_naive(Q2, K2, V2)
for bs in [32, 64, 128, 256]:
    with timer(f"Flash block={bs}"):
        o2, _, _ = attention_flash(Q2, K2, V2, block_size=bs)
    print(f"    与朴素版的最大误差 = {(o1 - o2).abs().max().item():.2e}")
print("""
  块大小的选择是典型工程权衡（第 7 章会亲手调）：
    块太小 → kernel 启动/循环次数多，但占用 shared memory 少
    块太大 → shared memory / 寄存器不够，occupancy 下降
  典型值：64 或 128（取决于 head_dim 与 GPU 架构）
""")

section("4) 反向传播怎么办？重计算（recompute）")
print("""
  FlashAttention 不保存 T×T 的 P 矩阵，那反向时 softmax 的梯度从哪来？
    → 前向时只存 O、m、l（O(T)），反向时用它们【重算】分块内的 P（再算一遍 QKᵀ）
  这是一个"用算力换显存"的经典交易：
    · 朴素：前向算 1 次、反向读 1 次（多花显存）
    · Flash：前向算 1 次、反向再算 1 次（多花 ~30% 算力，省 O(T²) 显存）
  在 GPU 上"重算"通常比"从 HBM 读"更快 —— 这也是第 7 章反复出现的主题：
    现代 GPU 是【访存受限】的，算力比带宽便宜。
""")

section("5) 从算法到落地")
print("""
  · PyTorch 2.x：F.scaled_dot_product_attention 会自动选择
      flash（Ampere+）/ mem-efficient（通用）/ math（兜底）三种后端
  · 直接 pip 安装 flash-attn 可拿到手写 CUDA 版，支持 GQA、变长序列、滑动窗口
  · 相关演进：FlashAttention-2（更好的并行与 warp 划分）、FlashAttention-3（Hopper，
     用 TMA/WGMMA 异步流水线 + fp8）
  · 别自己写生产用的注意力 kernel，除非你要做研究；但一定要懂它的原理 ——
     因为"KV Cache 分页""长上下文""稀疏注意力"全都建立在这套分块思想上。
""")
