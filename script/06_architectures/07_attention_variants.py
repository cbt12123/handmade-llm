"""
07 · 注意力变体大乱斗：MHA / MQA / GQA / MLA / 滑窗 / ALiBi
============================================================
这一节回答一个问题：**同样效果下，怎么让 KV Cache 更小、注意力更快？**

  MHA  多查询头 + 多 KV 头（1:1）            显存最大，质量最好
  MQA  多查询头 + 1 个 KV 头                  显存最小，质量略降（多用于小模型/端侧）
  GQA  多查询头 + 少量 KV 头（分组共享）       折中方案，现代 LLM 标配
  MLA  把 KV 压成低秩潜向量再缓存              DeepSeek 提出，比 GQA 还小
  滑窗 只看最近 W 个 token（配合多层堆叠扩大感受野）  Mistral 用
  ALiBi 不加位置编码，直接在分数上减距离惩罚       BLOOM / MPT 用

本脚本逐个实现并给出显存/效果对照表。
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import ascii_heatmap, bytes_str, require_torch, section, set_seed, subsection

set_seed(15)
torch = require_torch("07_attention_variants")
nn = torch.nn
F = torch.nn.functional

# ----------------------------------------------------------------------------------
# 以 Llama-3-8B 为基准
# ----------------------------------------------------------------------------------
LAYERS, D_MODEL, N_Q = 32, 4096, 32
D_HEAD = D_MODEL // N_Q


def kv_bytes_per_token(n_kv_heads, d_head=D_HEAD, layers=LAYERS, dtype_bytes=2):
    return layers * 2 * n_kv_heads * d_head * dtype_bytes


section("1) KV Cache 显存对照（32 层 / 32 个 Q 头 / head_dim=128 / fp16）")
VARIANTS = [
    ("MHA  (32 KV 头)", 32),
    ("GQA  (8 KV 头)  ← Llama-3-8B", 8),
    ("GQA  (4 KV 头)  ← Qwen2.5-7B", 4),
    ("GQA  (2 KV 头)  ← GLM-4-9B", 2),
    ("MQA  (1 KV 头)", 1),
]
print(f"  {'变体':<34}{'每 token':>12}{'8k 上下文':>12}{'128k 上下文':>14}{'相对 MHA':>10}")
base = kv_bytes_per_token(32)
for name, n_kv in VARIANTS:
    b = kv_bytes_per_token(n_kv)
    print(f"  {name:<34}{bytes_str(b):>12}{bytes_str(b * 8192):>12}"
          f"{bytes_str(b * 131072):>14}{b / base:>10.2f}")

section("2) MLA（Multi-head Latent Attention，DeepSeek 的核心创新）")
print("""
  MLA 的思路：K/V 本来就是 h_t 的线性变换，那干脆只缓存一个【压缩后的潜向量 c_t】

      c_t  = W_DKV · h_t                      维度 d_c（如 512）
      k_t  = W_UK · c_t  (+ 一小段 RoPE 分量)   需要时再解压缩
      v_t  = W_UV · c_t

  于是每个 token 只需缓存 d_c 维（+ RoPE 的 64 维），而不是 n_kv_heads × d_head 维。
  推理时 W_UK / W_UV 还可以【吸收】进 W_Q / W_O，连解压缩这一步都省了。
""")
D_C, D_ROPE = 512, 64
mla_bytes = LAYERS * (D_C + D_ROPE) * 2
gqa8_bytes = kv_bytes_per_token(8)
mha_bytes = kv_bytes_per_token(32)
print(f"  MLA  每 token: {bytes_str(mla_bytes)}   （d_c={D_C} + RoPE {D_ROPE}）")
print(f"  GQA-8 每 token: {bytes_str(gqa8_bytes)}")
print(f"  MHA   每 token: {bytes_str(mha_bytes)}")
print(f"  → MLA 相对 GQA-8 省 {(1 - mla_bytes / gqa8_bytes):.0%}，相对 MHA 省 {(1 - mla_bytes / mha_bytes):.0%}")


class MLAAttention(nn.Module):
    """极简 MLA：缓存低秩潜向量 c，需要时解压成 K/V。"""

    def __init__(self, d_model, n_heads, d_c=64, d_rope=16):
        super().__init__()
        self.h, self.dh = n_heads, d_model // n_heads
        self.d_c, self.d_rope = d_c, d_rope
        self.w_dkv = nn.Linear(d_model, d_c, bias=False)          # 下投影（压缩）
        self.w_uk = nn.Linear(d_c, n_heads * self.dh, bias=False)  # 上投影 → K
        self.w_uv = nn.Linear(d_c, n_heads * self.dh, bias=False)  # 上投影 → V
        self.w_q = nn.Linear(d_model, n_heads * self.dh, bias=False)
        self.w_o = nn.Linear(n_heads * self.dh, d_model, bias=False)

    def forward(self, x, is_causal=True):
        B, T, _ = x.shape
        c = self.w_dkv(x)                                          # ← 只有这个要缓存
        k = self.w_uk(c).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.w_uv(c).view(B, T, self.h, self.dh).transpose(1, 2)
        q = self.w_q(x).view(B, T, self.h, self.dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (self.dh ** 0.5)
        if is_causal:
            scores = scores.masked_fill(
                ~torch.ones(T, T, dtype=torch.bool, device=x.device).tril(), float("-inf"))
        w = torch.softmax(scores, dim=-1)
        out = (w @ v).transpose(1, 2).reshape(B, T, -1)
        return self.w_o(out), c

    def cache_bytes_per_token(self, dtype_bytes=2):
        return self.d_c * dtype_bytes


mla = MLAAttention(256, 8, d_c=64, d_rope=16)
x = torch.randn(2, 12, 256)
out, c = mla(x)
print(f"\n  玩具 MLA: 输入 {tuple(x.shape)} → 输出 {tuple(out.shape)}，缓存 c 的形状 {tuple(c.shape)}")
print(f"  缓存量 = {mla.cache_bytes_per_token()} 字节/token/层"
      f"（同等 MHA 需要 {2 * 8 * 32 * 2} 字节）")

section("3) 滑动窗口注意力（Mistral 的 4096 窗口）")
def sliding_window_mask(T, W):
    i = np.arange(T)[:, None]
    j = np.arange(T)[None, :]
    return (np.abs(i - j) <= W) & (j <= i)            # 只看左边 W 个


T_DEMO, W = 24, 4
sw = sliding_window_mask(T_DEMO, W)
print(ascii_heatmap(sw.astype(float), width=48, height=14, title="  滑动窗口掩码（W=4，行=query，列=key）："))
print(f"""
  单层感受野 = W+1 = {W + 1}；堆叠 L 层后感受野 = 1 + L·W
    Mistral-7B: W=4096, L=32 → 理论感受野 {1 + 32 * 4096:,} token（远超其 32k 上下文）
  优点：注意力计算从 O(T²) 降到 O(T·W)；KV Cache 只需保留窗口内
  注意：滑窗层要与【全局注意力层】交错使用，否则远距离信息传不过来
""")

section("4) ALiBi：不加位置编码，直接在分数上做距离惩罚")
def alibi_bias(T, n_heads, slopes=None):
    """scores += -slope_h · |i − j|；不同头用不同斜率（几何级数）。"""
    if slopes is None:
        slopes = 2 ** (-8 * np.arange(1, n_heads + 1) / n_heads)
    i = np.arange(T)[:, None]
    j = np.arange(T)[None, :]
    dist = np.abs(i - j)
    return -slopes[:, None, None] * dist[None]           # (h, T, T)


bias = alibi_bias(16, 4)
print(f"  4 个头的斜率: {np.round(2 ** (-8 * np.arange(1, 5) / 4), 4)}")
print(ascii_heatmap(bias[0], width=44, height=12, title="  头 0 的 ALiBi 偏置（越远惩罚越大）："))
print("""
  优点：外推能力极好（惩罚是线性的，超出训练长度也能合理工作）、零额外参数
  缺点：不如 RoPE 灵活（RoPE 能表达更复杂的位置关系）
  现状：多数新模型选 RoPE，但 ALiBi 的理念（相对位置惩罚）被吸收进了
        各种 "注意力 logit 修正" 的技巧里（如长度惩罚、位置插值）
""")

section("5) 怎么选")
print("""
  · 追求质量、显存充足         → MHA（现在很少用，推理太贵）
  · 通用大模型默认             → GQA（8 或 4 个 KV 头，几乎不掉点）
  · 极致压缩显存（端侧/长文）  → MQA 或 MLA
  · 长上下文（>64k）           → GQA/MLA + 滑窗 + 长上下文继续训练
  · 从零训练小模型             → 直接上 GQA，别先训 MHA 再改（转换会掉点）
  记住一条主线：所有变体的目标都是【在不掉点的前提下，把 KV Cache 变小】，
  因为 LLM 推理的瓶颈从来不是算力，而是"把 KV 从显存里读出来"的时间。
""")
