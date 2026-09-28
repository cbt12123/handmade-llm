"""
01 · 缩放点积注意力（Scaled Dot-Product Attention）从零手写
==============================================================
一句话公式：

        Attention(Q, K, V) = softmax( QKᵀ / √d_k ) · V

拆解成 4 步（就是 4 行 numpy）：
    ① 打分 scores = Q @ Kᵀ            每个 query 与所有 key 的相似度
    ② 缩放 scores /= √d_k             防止 softmax 饱和（第 3 章第 6 节讲过）
    ③ 归一化 weights = softmax(scores) 转成"注意力分配比例"
    ④ 聚合 output = weights @ V        按权重把 value 加权平均

本脚本：手写 → 数值示例 → 与 torch 官方实现对照 → 复杂度分析。
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import ascii_heatmap, check_close, section, set_seed, subsection

set_seed(0)


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def attention(Q, K, V, mask=None):
    """Q:(Tq,d) K:(Tk,d) V:(Tk,dv) → out:(Tq,dv), weights:(Tq,Tk)"""
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)              # ①②
    if mask is not None:
        scores = np.where(mask, scores, -1e9)    # 被 mask 的位置填 −∞（用 −1e9 代替，避免 nan）
    weights = softmax(scores, axis=-1)           # ③
    return weights @ V, weights                  # ④


# ----------------------------------------------------------------------------------
section("1) 一个手算得清的小例子：4 个 token，维度 d=4")
T, D = 4, 4
rng = np.random.default_rng(0)
X = rng.normal(size=(T, D))                     # 词嵌入（先假装没有位置编码）
# 自注意力：Q/K/V 都由同一个 X 线性变换而来
W_Q = rng.normal(0, 0.5, (D, D))
W_K = rng.normal(0, 0.5, (D, D))
W_V = rng.normal(0, 0.5, (D, D))
Q, K, V = X @ W_Q, X @ W_K, X @ W_V

out, w = attention(Q, K, V)
print(f"  输入 X 形状 {X.shape}")
print(f"  Q/K/V 形状 {Q.shape} {K.shape} {V.shape}")
print(f"  注意力权重形状 {w.shape}（每一行和为 1: {np.round(w.sum(1), 6)}）")
print(ascii_heatmap(w, width=44, height=10, title="  注意力权重矩阵（行=query 位置, 列=key 位置）："))
print(f"  输出形状 {out.shape}")

section("2) 逐行验证：输出确实等于 V 的加权平均")
i = 2
manual = np.zeros(D)
for j in range(T):
    manual += w[i, j] * V[j]
check_close(out[i], manual, tol=1e-12, name=f"第 {i} 个输出 = Σⱼ αᵢⱼ·Vⱼ")
print(f"  第 {i} 个 token 的注意力分配：{np.round(w[i], 4)}")
print(f"  它主要『看』的是第 {int(np.argmax(w[i]))} 个 token。")

section("3) 和 PyTorch 官方实现对照")
try:
    import torch
    import torch.nn.functional as F

    Qt = torch.tensor(Q[None], dtype=torch.float32)
    Kt = torch.tensor(K[None], dtype=torch.float32)
    Vt = torch.tensor(V[None], dtype=torch.float32)
    official = F.scaled_dot_product_attention(Qt, Kt, Vt)
    check_close(out[None], official[0], tol=1e-5, name="手写 attention vs torch SDPA")
    print("  torch 的 scaled_dot_product_attention 在 GPU 上会自动调用 FlashAttention 内核")
    print("  （第 5、7 章会讲它为什么快：不落地完整的 T×T 矩阵）")
except ImportError:
    print("[SKIP] 未安装 torch，跳过对照")

section("4) 自注意力 vs 交叉注意力")
X2 = rng.normal(size=(6, D))                     # 另一句话（6 个 token）
Kc, Vc = X2 @ W_K, X2 @ W_V
out_cross, w_cross = attention(Q, Kc, Vc)
print(f"  自注意力  : Q 来自 X(T=4)，K/V 也来自 X   → 权重 {w.shape}")
print(f"  交叉注意力: Q 来自 X(T=4)，K/V 来自 X2(T=6) → 权重 {w_cross.shape}")
print(ascii_heatmap(w_cross, width=44, height=10, title="  交叉注意力权重（4×6）："))
print("  用法：解码器的交叉注意力层（看编码器输出），也就是第 3 章 seq2seq attention 的位置。")

section("5) 复杂度：注意力为什么『贵』")
for t in [512, 2048, 8192, 32768, 131072]:
    n_elem = t * t
    print(f"  T={t:>7}  注意力矩阵元素 = {n_elem:>12,}  "
          f"fp16 显存 ≈ {n_elem * 2 / 1024**2:>8.1f} MB（单头单样本）")
print("""
  · 时间 O(T²·d)、显存 O(T²) → 这正是"长上下文"最大的工程挑战
  · 第 5 章的 KV Cache / FlashAttention / 滑动窗口，第 6 章的 Mamba / Linear Attention
    本质上都是在想办法绕过这个 T²
""")

section("6) 一句话总结")
print("""
  注意力 = 可微分的"字典查询"：
      Q 是查询词，K 是索引，V 是内容，softmax 是"软化的 argmax"
  它取代 RNN 的关键不是"效果更好"，而是"可以被写成一次矩阵乘法" → 完全并行 → GPU 友好。
""")
