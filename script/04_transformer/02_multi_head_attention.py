"""
02 · 多头注意力（Multi-Head Attention, MHA）
==============================================
为什么要"多头"？
  单个注意力头只能表达一种"相关性"；多头 = 在不同子空间里并行地做多次注意力，
  让模型同时捕捉"语法依赖""指代关系""位置邻近"等多种模式。

公式（d_model = h · d_head）：
    head_i = Attention(X·W_Qⁱ, X·W_Kⁱ, X·W_Vⁱ)
    MHA(X) = Concat(head_1, ..., head_h) · W_O

关键点：多头注意力的总计算量与单头（d_model 维）几乎相同 —— 因为每个头的维度是 d_model/h。
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import ascii_heatmap, check_close, human_num, section, set_seed, subsection

set_seed(1)


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def split_heads(x, h):
    """(T, d_model) → (h, T, d_head)"""
    T, D = x.shape
    return x.reshape(T, h, D // h).transpose(1, 0, 2)


def merge_heads(x):
    """(h, T, d_head) → (T, d_model)"""
    h, T, dh = x.shape
    return x.transpose(1, 0, 2).reshape(T, h * dh)


class MultiHeadAttention:
    def __init__(self, d_model, num_heads, seed=0):
        rng = np.random.default_rng(seed)
        self.h, self.d_model = num_heads, d_model
        self.d_head = d_model // num_heads
        s = 0.4
        self.W_Q = rng.normal(0, s, (d_model, d_model))
        self.W_K = rng.normal(0, s, (d_model, d_model))
        self.W_V = rng.normal(0, s, (d_model, d_model))
        self.W_O = rng.normal(0, s, (d_model, d_model))

    def __call__(self, X, mask=None, return_weights=False):
        Q = split_heads(X @ self.W_Q, self.h)
        K = split_heads(X @ self.W_K, self.h)
        V = split_heads(X @ self.W_V, self.h)
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.d_head)      # (h, T, T)
        if mask is not None:
            scores = np.where(mask, scores, -1e9)
        w = softmax(scores, axis=-1)
        ctx = w @ V                                                   # (h, T, d_head)
        out = merge_heads(ctx) @ self.W_O
        return (out, w) if return_weights else out


# ----------------------------------------------------------------------------------
section("1) 形状变化全过程")
T, D, H = 6, 16, 4
X = np.random.default_rng(0).normal(size=(T, D))
mha = MultiHeadAttention(D, H)
out, w = mha(X, return_weights=True)
print(f"  输入 X           : {X.shape}")
print(f"  Q/K/V 分头后     : (h, T, d_head) = ({H}, {T}, {D // H})")
print(f"  注意力权重 w     : {w.shape}  ← 每个头一张 T×T 的注意力图")
print(f"  拼接后           : ({T}, {D})")
print(f"  输出 out         : {out.shape}  ← 输入输出同形，所以可以堆叠 N 层")

section("2) 不同头关注不同模式（随机初始化下也能看出差异）")
for i in range(H):
    print(f"\n  头 #{i} 的注意力图：")
    print(ascii_heatmap(w[i], width=40, height=10))
print("\n  真实训练后，你会看到这样的分工：")
print("   头 0：关注前一个词（局部语法）   头 1：关注句首 <cls>（全局聚合）")
print("   头 2：关注同类型的词（共指/搭配） 头 3：关注标点/分隔符")

section("3) 参数量与计算量")
params = 4 * D * D                                  # W_Q, W_K, W_V, W_O
print(f"  MHA 参数量 = 4·d_model² = {params}（与头数无关！）")
print(f"  计算量     = O(T²·d_model)：分头只是把 d_model 拆成 h 份，总量不变")
print("  所以『多头』几乎是免费的 —— 这也是它成为标配的原因。")

section("4) 与 PyTorch nn.MultiheadAttention 对照")
try:
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    m = nn.MultiheadAttention(D, H, batch_first=True, bias=False)
    with torch.no_grad():
        # 把我们手写的权重拷进去，保证两者是同一个函数
        m.in_proj_weight.copy_(torch.cat([
            torch.tensor(mha.W_Q.T, dtype=torch.float32),
            torch.tensor(mha.W_K.T, dtype=torch.float32),
            torch.tensor(mha.W_V.T, dtype=torch.float32)]))
        m.out_proj.weight.copy_(torch.tensor(mha.W_O.T, dtype=torch.float32))
        y_torch, w_torch = m(torch.tensor(X[None], dtype=torch.float32),
                             torch.tensor(X[None], dtype=torch.float32),
                             torch.tensor(X[None], dtype=torch.float32),
                             need_weights=True, average_attn_weights=False)
    check_close(out, y_torch[0], tol=1e-4, name="手写 MHA vs torch nn.MultiheadAttention")
    check_close(w, w_torch[0], tol=1e-5, name="注意力权重 手写 vs torch")
except ImportError:
    print("[SKIP] 未安装 torch，跳过对照")

section("5) 工程细节（面试 & 实战高频）")
print(f"""
  ① 为什么要 W_O？ 多头拼接后需要一次线性混合，否则各头信息永远不交流
  ② d_model 必须能被 h 整除：Llama-3 8B = 32 头 × 128 维 = 4096
  ③ GQA/MQA（第 6 章）：让多个 query 头共享同一份 K/V → KV Cache 显存降到 1/8 ~ 1/32
  ④ 注意力的 FLOPs 占比：在 7B 模型、T=4096 时，注意力约占总计算的 20%~40%，
     其余是 FFN 的矩阵乘法
  ⑤ 参数量速算：Llama-7B 的 MHA 部分 ≈ 层数 × 4·d² = 32 × 4 × 4096² ≈ {human_num(32 * 4 * 4096.0**2)}
""")
