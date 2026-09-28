"""
02 · Encoder-Decoder 代表：T5 / BART
======================================
T5 的口号："Everything is Text-to-Text"（所有任务都改写成"输入文本 → 输出文本"）
  翻译：  "translate English to German: That is good." → "Das ist gut."
  分类：  "cola sentence: The course is jumping well."  → "acceptable"
  摘要：  "summarize: ..."                              → "..."

T5 与原版 Transformer 的 4 个关键差异：
  ① RMSNorm（不是 LayerNorm，去掉了均值与偏置，更快更稳）
  ② 相对位置偏置（不是位置编码：直接在注意力分数上加一个可学习的 bias）
  ③ 所有 Linear 都不带 bias
  ④ 预训练目标是 Span Corruption（把一段文本随机挖掉，用哨兵 token 标记，让模型还原）

本脚本手写 T5 block，并把"相对位置偏置"和"哨兵 token"这两个最有特色的设计跑一遍。
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import ascii_heatmap, human_num, require_torch, section, set_seed, subsection

set_seed(5)
torch = require_torch("02_encoder_decoder_t5")
nn = torch.nn
F = torch.nn.functional


# ----------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """T5 / Llama 用的归一化：只除以 RMS，不减均值，也没有可学习的 bias。"""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    """T5 的"分桶"技巧：把相对位置 [-max, max] 映射到 num_buckets 个桶。

    小距离用精确桶（每 1 个位置一个桶），大距离用对数桶（越远越粗）。
    """
    ret = 0
    if bidirectional:
        num_buckets //= 2
        ret += (relative_position > 0).long() * num_buckets
        n = torch.abs(relative_position)
    else:
        n = torch.max(-relative_position, torch.zeros_like(relative_position))
    max_exact = num_buckets // 2
    is_small = n < max_exact
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact) / np.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).long()
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
    return ret + torch.where(is_small, n, val_if_large)


class T5Attention(nn.Module):
    """带相对位置偏置的多头注意力（自注意力和交叉注意力共用）。"""

    def __init__(self, d_model, heads, has_relative_bias=True, num_buckets=32):
        super().__init__()
        self.h, self.dh = heads, d_model // heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.has_relative_bias = has_relative_bias
        if has_relative_bias:
            self.rel_bias = nn.Embedding(num_buckets, heads)

    def forward(self, x, memory=None, is_causal=False):
        B, Tq, _ = x.shape
        kv = x if memory is None else memory
        Tk = kv.shape[1]
        q = self.q(x).view(B, Tq, self.h, self.dh).transpose(1, 2)
        k = self.k(kv).view(B, Tk, self.h, self.dh).transpose(1, 2)
        v = self.v(kv).view(B, Tk, self.h, self.dh).transpose(1, 2)
        scores = q @ k.transpose(-2, -1)                       # ← 注意：T5 不除以 √d！
        if self.has_relative_bias:
            ctx = torch.arange(Tq, device=x.device)[:, None] - torch.arange(Tk, device=x.device)[None, :]
            buckets = relative_position_bucket(ctx).to(x.device)
            bias = self.rel_bias(buckets).permute(2, 0, 1)     # (h, Tq, Tk)
            scores = scores + bias
        if is_causal:
            scores = scores.masked_fill(~torch.ones(Tq, Tk, dtype=torch.bool, device=x.device).tril(),
                                        float("-inf"))
        w = torch.softmax(scores, dim=-1)
        out = (w @ v).transpose(1, 2).reshape(B, Tq, -1)
        return self.o(out), w


class T5Block(nn.Module):
    def __init__(self, d_model, heads, d_ff, has_relative_bias=True, cross_attn=False):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = T5Attention(d_model, heads, has_relative_bias)
        self.cross = None
        if cross_attn:
            self.ln_cross = RMSNorm(d_model)
            self.cross = T5Attention(d_model, heads, has_relative_bias=False)   # 交叉注意力无相对位置
        self.ln2 = RMSNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=False), nn.ReLU(),
                                nn.Linear(d_ff, d_model, bias=False))

    def forward(self, x, memory=None, is_causal=False):
        h = self.ln1(x)
        o, w = self.attn(h, is_causal=is_causal)
        x = x + o
        if self.cross is not None and memory is not None:
            x = x + self.cross(self.ln_cross(x), memory=memory)[0]
        x = x + self.ff(self.ln2(x))
        return x, w


section("1) RMSNorm vs LayerNorm")
d = 8
x = torch.randn(2, 5, d) * 3 + 1.5
ln = nn.LayerNorm(d)
rms = RMSNorm(d)
print(f"  输入均值={x.mean():.3f}  标准差={x.std():.3f}")
print(f"  LayerNorm 输出 均值={ln(x).mean():.3f} 标准差={ln(x).std():.3f}")
print(f"  RMSNorm   输出 均值={rms(x).mean():.3f} 标准差={rms(x).std():.3f}")
print("""
  差异：RMSNorm 不做"减均值"（re-centering），只做缩放（re-scaling）
  好处：少一次均值归约（更快）、数值更稳定、参数更少
  现在几乎所有 LLM（Llama / Qwen / Mistral / DeepSeek）都用 RMSNorm。
""")

section("2) 相对位置偏置的分桶机制")
mem_pos = torch.arange(-20, 21)
buckets = relative_position_bucket(mem_pos, num_buckets=32, max_distance=128)
print("  相对位置: ", " ".join(f"{v:>3}" for v in mem_pos.tolist()[:21]))
print("  对应桶号: ", " ".join(f"{v:>3}" for v in buckets.tolist()[:21]))
print("""
  近处（|d| < 16）每个位置一个桶 → 精确建模局部顺序
  远处（|d| ≥ 16）用对数分桶   → 远距离共享参数，泛化到训练外的长度
  这就是 T5 不需要位置编码却仍能感知顺序的原因。
""")

section("3) 组装一个 T5（encoder-decoder）并跑一次前向")
D_MODEL, H, D_FF = 64, 4, 128
enc = nn.ModuleList([T5Block(D_MODEL, H, D_FF) for _ in range(2)])
dec = nn.ModuleList([T5Block(D_MODEL, H, D_FF, cross_attn=True) for _ in range(2)])
emb = nn.Embedding(100, D_MODEL)
head = nn.Linear(D_MODEL, 100, bias=False)
print(f"  单层 encoder 参数 = {human_num(float(sum(p.numel() for p in enc[0].parameters())))}")
print(f"  单层 decoder 参数 = {human_num(float(sum(p.numel() for p in dec[0].parameters())))}（多一个交叉注意力）")

src = torch.randint(0, 100, (2, 10))
tgt = torch.randint(0, 100, (2, 6))
memory = emb(src)
for blk in enc:
    memory, _ = blk(memory)
y = emb(tgt)
for blk in dec:
    y, w = blk(y, memory=memory, is_causal=True)
logits = head(y)
print(f"  src {tuple(src.shape)} → memory {tuple(memory.shape)} → logits {tuple(logits.shape)}")

section("4) Span Corruption：T5 的预训练目标")
print("""
  原句：   "Thank you for inviting me to your party last week ."
  输入：   "Thank you <X> me to your party <Y> week ."
  目标：   "<X> for inviting <Y> last <Z>"
  做法：随机选 15% 的 token，把连续片段（span）替换成一个哨兵 token（<extra_id_0> 等）

  为什么比 BERT 的 MLM 更好：
    · 预测的是"一段话"而不是"一个词" → 学习信号更密集、更接近生成任务
    · 天然训练了 decoder 的生成能力 → 迁移到翻译/摘要更顺滑
""")
sentence = "Thank you for inviting me to your party last week".split()
masked = ["Thank", "you", "<X>", "me", "to", "your", "party", "<Y>", "week"]
target = ["<X>", "for", "inviting", "<Y>", "last"]
print(f"  示例输入: {' '.join(masked)}")
print(f"  示例目标: {' '.join(target)}  （+ 结尾哨兵 <Z>）")

section("5) Encoder-Decoder 的今天")
print("""
  优势：
    · 编码器只需要算一次（不像 decoder-only 每步都要看全部上下文）→ 长输入更高效
    · 翻译/摘要这类"输入-输出强对应"的任务上依然是最优结构
  劣势：
    · 参数利用率低：一份参数只能干"生成"这一件事（encoder 部分不参与生成）
    · 通用对话/推理任务上，decoder-only 的规模化收益更明显
  现状：
    · 2023 年后新发布的通用 LLM 几乎全是 decoder-only
    · 但 encoder-decoder 在【语音（Whisper）、翻译（NLLB）、多模态】上仍是首选
    · Whisper 就是典型的 encoder-decoder Transformer，至今无可替代
""")
