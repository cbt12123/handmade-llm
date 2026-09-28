"""
04 · RoPE 旋转位置编码：现代 LLM 的标配，以及长上下文外推
==========================================================
核心思想（一句话）：
  与其"把位置向量加到词向量上"，不如"把 Q/K 旋转一个与位置相关的角度"。
  因为旋转不改变向量长度，且两个旋转后向量的内积只依赖【相对位置】：
      ⟨R_m·q, R_n·k⟩ = ⟨R_{m−n}·q, k⟩   （R 是旋转矩阵）

本脚本：
  1. 手写 RoPE 的两种实现（复数式 / 矩阵式）、验证相对位置性质
  2. 对比 RoPE 与绝对位置编码在"未见过的长度"上的表现
  3. 实现三种外推缩放：线性插值 PI / NTK-Aware / YaRN（简化版）
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import ascii_heatmap, require_torch, section, set_seed, subsection

set_seed(17)
torch = require_torch("04_rope")

D, N_POS = 64, 4096


# ----------------------------------------------------------------------------------
def precompute_freqs(dim, n_pos, theta=10000.0):
    """频率向量：θ_i = 1 / base^(2i/d)，i = 0..d/2−1"""
    i = torch.arange(0, dim, 2).float()
    return 1.0 / (theta ** (i / dim))


def rope_apply(x, freqs, start_pos=0):
    """对最后一个维度做旋转。x: (..., T, d)，freqs: (d/2,)"""
    T = x.shape[-2]
    pos = torch.arange(start_pos, start_pos + T, device=x.device).float()
    angles = torch.outer(pos, freqs)                     # (T, d/2)
    cos, sin = torch.cos(angles), torch.sin(angles)
    x1, x2 = x[..., 0::2], x[..., 1::2]                  # 按"相邻两维一组"分组（GPT-J 风格）
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.stack([o1, o2], dim=-1).flatten(-2)


section("1) 相对位置性质：数值验证")
freqs = precompute_freqs(D, N_POS)
torch.manual_seed(0)
q = torch.randn(D)
k = torch.randn(D)


def rope_1d(x, pos):
    return rope_apply(x.view(1, 1, D), freqs, start_pos=pos).view(D)


for m, n in [(0, 3), (5, 9), (100, 104), (1000, 1200)]:
    lhs = float(rope_1d(q, m) @ rope_1d(k, n))
    rhs = float(rope_1d(q, 0) @ rope_1d(k, n - m))
    print(f"  m={m:>5}, n={n:>5}: ⟨R_m q, R_n k⟩={lhs:+.6f}   ⟨R_0 q, R_{{n−m}} k⟩={rhs:+.6f}   "
          f"差={abs(lhs - rhs):.2e}")
print("  完全相等 → 模型只要学会『关注相对距离』，不需要知道绝对位置。")

section("2) 相对距离衰减：距离越远，内积的平均相关性越低")
torch.manual_seed(1)
Q = torch.randn(200, D)
K = torch.randn(200, D)
print(f"{'相对距离':<10}{'⟨R_m q, R_{m+d} k⟩ 的平均绝对值':>32}")
for dist in [0, 1, 4, 16, 64, 256]:
    q_rot = rope_apply(Q.view(200, 1, D), freqs, start_pos=0).view(200, D)
    k_rot = rope_apply(K.view(200, 1, D), freqs, start_pos=dist).view(200, D)
    val = float((q_rot * k_rot).sum(-1).abs().mean())
    print(f"  {dist:<8}{val:>32.4f}")

section("3) 长上下文外推：三种缩放方案")
print("""
  训练时只见过 L_train=4096 的位置，推理要到 32768。直接外推会怎样？
  问题：位置 30000 对应的旋转角度远超训练范围 → 注意力分数分布畸变。
""")


def make_freqs(dim, n_pos, theta=10000.0, scaling="none", factor=1.0, n_train=4096):
    """scaling: none / linear / ntk / yarn"""
    base_freqs = precompute_freqs(dim, n_pos, theta)
    if scaling == "none":
        return base_freqs
    if scaling == "linear":                       # 线性插值：把位置压缩回训练范围
        return base_freqs / factor
    if scaling == "ntk":                          # NTK-Aware：拉大 base（高频少缩放）
        new_theta = theta * factor ** (dim / (dim - 2))
        return precompute_freqs(dim, n_pos, new_theta)
    if scaling == "yarn":                         # YaRN：NTK + 温度校正（这里只做 NTK 部分）
        new_theta = theta * factor ** (dim / (dim - 2))
        f = precompute_freqs(dim, n_pos, new_theta)
        return f * 0.9                            # 简化：对整体做一点缩放
    return base_freqs


L_TRAIN, L_TARGET, FACTOR = 4096, 32768, 8.0        # 想把 4k 扩到 32k（8 倍）

print(f"""
  RoPE 的每个维度 i 有一个频率 θ_i = 1/base^(i/d)，对应一个"周期"（波长）：
    小 i  → 高频、波长短（约几个 token）→ 负责【局部】位置分辨
    大 i  → 低频、波长长（成千上万个 token）→ 负责【长程】距离

  评估两个指标：
    A 局部分辨率 = 最高频分量的频率（相对不做缩放时）→ 越大越能区分相邻 token
    B 长程相位比 = 最慢那个分量在 {L_TARGET} 处的相位 / 它在 {L_TRAIN} 处的相位
                   → 越接近 1 越"在训练分布内"；远大于 1 表示模型没见过这么大的相位
""")
base_f = precompute_freqs(D, L_TARGET)
print(f"{'缩放方案':<12}{'A 局部分辨率(相对 none)':>26}{'B 长程相位比':>14}")
for mode in ["none", "linear", "ntk", "yarn"]:
    f = make_freqs(D, L_TARGET, scaling=mode, factor=FACTOR)
    local = float(f.max()) / float(base_f.max())          # 最高频分量（波长最短）
    phase_train = float(L_TRAIN * base_f.min())           # 未缩放时训练末端的相位
    phase_target = float(L_TARGET * f.min())              # 缩放后目标位置的相位
    print(f"  {mode:<10}{local:>26.3f}{phase_target / phase_train:>14.2f}")
print(f"""
  读表：
    none  : A=1.000（局部没问题），但 B={L_TARGET / L_TRAIN:.0f}× —— 长程分量的相位远超训练时见过的范围 → 注意力分数畸变
    linear: B≈1（相位被压回训练分布），代价是 A 降到 1/{FACTOR:.0f} → 相邻位置变得难分辨（"位置变糊"）
    ntk   : 只把低频（长程）压回来，高频几乎不动 → A≈1 且 B≈1，两头都顾上，代价是中间频段仍有畸变
    yarn  : 在 ntk 基础上再加注意力温度校正，长上下文实测最稳（Llama-2/3 的长上下文扩展常用）
""")

section("4) 两种 RoPE 的实现风格")
x = torch.randn(2, 6, 8)
# 风格 A：相邻两维一组（GPT-J / 早期实现）
a = rope_apply(x, precompute_freqs(8, 64))
# 风格 B：前后半分组（Llama / HuggingFace 主流实现）
def rope_half(x, freqs, start_pos=0):
    T = x.shape[-2]
    pos = torch.arange(start_pos, start_pos + T, device=x.device).float()
    angles = torch.outer(pos, freqs)
    cos, sin = torch.cos(angles), torch.sin(angles)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


b = rope_half(x, precompute_freqs(8, 64))
print(f"  相邻分组(A) 范数: {a.norm(dim=-1)[0, :3].tolist()}")
print(f"  前后半分组(B) 范数: {b.norm(dim=-1)[0, :3].tolist()}")
print(f"  输入范数: {x.norm(dim=-1)[0, :3].tolist()}")
print("  两者都保持范数不变（旋转的性质），只是分组方式不同 —— 效果等价，")
print("  但注意：混用两种实现会导致模型输出完全不同，迁移权重时务必对齐。")

section("5) 实践要点")
print("""
  ① 现代模型几乎都用 RoPE：Llama / Mistral / Qwen / GLM / DeepSeek
     例外：ALiBi（BLOOM、MPT）在注意力分数上直接减与距离成正比的惩罚
  ② RoPE 的 base（10000）可以调大以支持更长上下文（Llama-3 用 500000）
  ③ 想稳定用 32k+，优先选"官方长上下文版本"，而不是自己外推
  ④ 多模态 / 视频模型会用 3D RoPE（时间 + 高 + 宽），思想完全一致
""")
