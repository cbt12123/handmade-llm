"""
04 · 现代 Decoder-only 的"标准件"：Llama 家族
==============================================
从 GPT-2（2019）到 Llama（2023），结构层面只改了 5 件事，但每一件都被沿用至今：

  ① LayerNorm → RMSNorm                     更快、更省、更稳
  ② 绝对位置编码 → RoPE                      支持相对位置与长度外推
  ③ ReLU/GELU → SwiGLU                      同样的参数量下效果更好
  ④ MHA → GQA（分组查询注意力）               KV Cache 直接减半甚至更多
  ⑤ 去掉所有 Linear 的 bias + Pre-LN         训练更稳、算子更快

本脚本逐个实现这些组件，并量化它们带来的收益。
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import bytes_str, human_num, require_torch, section, set_seed, subsection

set_seed(9)
torch = require_torch("04_llama_family")
nn = torch.nn
F = torch.nn.functional


# ----------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope(x, freqs):
    """x: (B, T, h, dh) —— 按"前后半分组"旋转（Llama 的实现方式）。"""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos, sin = torch.cos(freqs), torch.sin(freqs)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class SwiGLUFFN(nn.Module):
    """FFN_SwiGLU(x) = (Swish(xW₁) ⊙ xW₂) W₃     —— 三个矩阵，没有 bias"""

    def __init__(self, d_model, d_ff=None, multiple_of=256):
        super().__init__()
        d_ff = d_ff or int(8 * d_model / 3)
        d_ff = int(multiple_of * ((d_ff + multiple_of - 1) // multiple_of))
        self.w1 = nn.Linear(d_model, d_ff, bias=False)      # gate
        self.w2 = nn.Linear(d_model, d_ff, bias=False)      # up
        self.w3 = nn.Linear(d_ff, d_model, bias=False)      # down
        self.d_ff = d_ff

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))     # silu(x) = x·σ(x) = Swish


def repeat_kv(x, n_rep):
    """GQA：把 KV 头复制 n_rep 份，使之与 Q 的头数对齐（实现上常用 view/expand，不真复制）。"""
    if n_rep == 1:
        return x
    B, H, T, D = x.shape
    return x[:, :, None, :, :].expand(B, H, n_rep, T, D).reshape(B, H * n_rep, T, D)


class GroupedQueryAttention(nn.Module):
    """n_kv_heads < n_heads 即为 GQA；n_kv_heads == 1 即为 MQA。"""

    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.dh = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.dh, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.dh, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, is_causal=True):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.dh).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.dh).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.dh).transpose(1, 2)
        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)
        scores = (q @ k.transpose(-2, -1)) / (self.dh ** 0.5)
        if is_causal:
            scores = scores.masked_fill(
                ~torch.ones(T, T, dtype=torch.bool, device=x.device).tril(), float("-inf"))
        w = torch.softmax(scores, dim=-1)
        out = (w @ v).transpose(1, 2).reshape(B, T, -1)
        return self.wo(out)


class LlamaBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, norm_eps=1e-6):
        super().__init__()
        self.ln1 = RMSNorm(d_model, norm_eps)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
        self.ln2 = RMSNorm(d_model, norm_eps)
        self.ff = SwiGLUFFN(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


section("1) SwiGLU vs 普通 GELU FFN：参数量与效果")
D = 4096
ffn_gelu = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
ffn_swiglu = SwiGLUFFN(D)
p_gelu = sum(p.numel() for p in ffn_gelu.parameters())
p_swiglu = sum(p.numel() for p in ffn_swiglu.parameters())
print(f"  GELU  FFN (4d)    : {human_num(float(p_gelu))} 参数，d_ff={4 * D}")
print(f"  SwiGLU FFN (8d/3) : {human_num(float(p_swiglu))} 参数，d_ff={ffn_swiglu.d_ff}")
print(f"  SwiGLU 多了 {p_swiglu / p_gelu - 1:.1%} 参数，但同参数预算下困惑度更低（PaLM/LLaMA 实测）")
print("  做法：把 d_ff 从 4d 降到 8d/3，参数量对齐，多出来的那一个矩阵用于『门控』。")

section("2) GQA：KV Cache 直接降一个数量级")
print(f"  {'配置':<24}{'K/V 参数':>14}{'KV Cache/token':>18}{'相对 MHA':>10}")
for name, n_kv in [("MHA (32 Q / 32 KV)", 32), ("GQA (32 Q / 8 KV)", 8),
                   ("GQA (32 Q / 4 KV)", 4), ("MQA (32 Q / 1 KV)", 1)]:
    gqa = GroupedQueryAttention(4096, 32, n_kv)
    kv_params = sum(p.numel() for p in [gqa.wk.weight, gqa.wv.weight])
    per_token = 2 * 32 * n_kv * 128 * 2            # 32 层 × K/V × 头 × head_dim × fp16
    print(f"  {name:<24}{human_num(float(kv_params)):>14}{bytes_str(per_token):>18}"
          f"{per_token / (2 * 32 * 32 * 128 * 2):>10.2f}")
print("""
  Llama-3-8B 用 8 个 KV 头（32 个 Q 头）→ KV Cache 是 MHA 的 1/4，几乎不损失效果
  Qwen2.5-7B 更激进：28 个 Q 头 / 4 个 KV 头 → 1/7
  注意：GQA 需要在训练时就这么训（或用 "GQA uptraining" 从 MHA 转换），
        直接把训好的 MHA 改成 GQA 会明显掉点。
""")

section("3) 参数量对比：GPT-2 风格 vs Llama 风格")
def count(blocks):
    return human_num(float(sum(p.numel() for p in blocks.parameters())))


gpt_block = nn.ModuleList([
    nn.LayerNorm(4096), nn.MultiheadAttention(4096, 32, batch_first=True),
    nn.Linear(4096, 4 * 4096), nn.GELU(), nn.Linear(4 * 4096, 4096), nn.LayerNorm(4096)])
llama_block = LlamaBlock(4096, 32, 8)
print(f"  GPT-2 风格 block : {count(gpt_block)} 参数")
print(f"  Llama 风格 block : {count(llama_block)} 参数")
print("  差异主要来自：SwiGLU 的 3 个矩阵 vs 2 个、GQA 的 K/V 变小、去掉 bias。")

section("4) 跑一次完整前向")
block = LlamaBlock(256, 8, 2)
x = torch.randn(2, 16, 256)
out = block(x)
print(f"  输入 {tuple(x.shape)} → 输出 {tuple(out.shape)}")
print(f"  RMSNorm 权重只有 {block.ln1.weight.numel()} 个参数（LayerNorm 是 2×d）")

section("5) 一张表记住现代 decoder-only 的标配")
print("""
  组件            GPT-2(2019)      Llama-3(2024)         为什么改
  ---------------------------------------------------------------------------
  归一化          LayerNorm(Pre)   RMSNorm(Pre)          更快、更稳、参数更少
  位置           绝对位置 Emb      RoPE(θ=500000)        相对位置 + 可外推
  FFN 激活       GELU             SwiGLU(8d/3)          同等参数下效果更好
  注意力         MHA              GQA(8 KV 头)          KV Cache 省 4~8 倍
  Linear bias    有               无                    少一次访存，训练更稳
  权重绑定       有               无                    大词表下解耦更灵活
  激活检查点     少用             训练必开               省激活显存
""")
