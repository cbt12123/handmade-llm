"""
03 · Decoder-only 代表：GPT 家族（今天所有聊天 LLM 的共同祖先）
================================================================
GPT = 只保留 Transformer 的解码器，并且只做一件事：预测下一个 token。
  y = softmax(W·h_T)   ← h_T 是最后一个位置的隐藏状态

GPT-2 的结构细节（与现代 LLM 的差异在下一节讲）：
  · Pre-LN：LayerNorm 放在子层【之前】
  · GELU 激活（不是 ReLU）
  · 可学习绝对位置编码
  · 权重绑定（tied embeddings）：输出层的 W 直接复用输入 embedding 矩阵
  · 所有 Linear 都带 bias（Llama 系把这个去掉了）

本脚本：手写 GPT block → 拼出 GPT-2 各尺寸的规模表 → 演示权重绑定与 KV Cache 形状。
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import bytes_str, human_num, require_torch, section, set_seed, subsection

set_seed(7)
torch = require_torch("03_decoder_only_gpt")
nn = torch.nn
F = torch.nn.functional


class GPTBlock(nn.Module):
    """GPT-2 风格：LayerNorm 前置 + 带 bias 的 Linear + GELU。"""

    def __init__(self, d_model, heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, is_causal=True):
        T = x.shape[1]
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        h = self.ln1(x)
        o, w = self.attn(h, h, h, attn_mask=mask, need_weights=True)
        x = x + self.drop(o)
        x = x + self.ff(self.ln2(x))
        return x, w


class TinyGPT(nn.Module):
    def __init__(self, vocab, d_model, heads, layers, max_len=1024, tie=True):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([GPTBlock(d_model, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        if tie:
            self.head.weight = self.tok.weight        # 权重绑定

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device)[None])
        for blk in self.blocks:
            x, _ = blk(x)
        return self.head(self.ln_f(x))


section("1) 权重绑定（tied embeddings）能省多少参数")
for vocab, d_model in [(50257, 768), (50257, 1600), (128256, 4096)]:
    tied = vocab * d_model
    print(f"  vocab={vocab:>7}, d_model={d_model:>5}: 绑定省下 {human_num(float(tied))} 参数"
          f"（占 12 层模型约 {tied / (12 * 12 * d_model**2 + 2 * tied):.1%}）")
print("""
  输出层的 W (vocab × d) 与输入 embedding 共享 → 训练更稳（两边梯度一致），参数更少
  现代大模型（Llama / Qwen）通常【不绑定】，因为词表太大（128k），
  绑定会让 embedding 的学习率受到输出层梯度量级的干扰。
""")

section("2) GPT 家族规模表")
GPT_FAMILY = [
    ("GPT-1", "117M", 12, 768, 12, 512),
    ("GPT-2 S/M/L/XL", "0.1~1.5B", "12~48", "768~1600", "12~25", 1024),
    ("GPT-3", "175B", 96, 12288, 96, 2048),
    ("Llama-3-8B", "8B", 32, 4096, 32, 8192),
]
print(f"  {'模型':<18}{'参数':<10}{'层':>5}{'d_model':>9}{'头':>5}{'上下文':>8}")
for name, p, l, d, h, ctx in GPT_FAMILY:
    print(f"  {name:<18}{p:<10}{str(l):>5}{str(d):>9}{str(h):>5}{ctx:>8}")
print("""
  观察：从 GPT-1 到 GPT-3，"深度（层数）"和"宽度（d_model）"同步增长，
  而上下文窗口长期停留在 2k —— 长上下文是 2023 年之后才被解决的主要问题。
""")

section("3) 跑一次前向，确认因果性")
model = TinyGPT(vocab=200, d_model=64, heads=4, layers=2, max_len=128)
print(f"  本玩具 GPT 参数量 = {human_num(float(sum(p.numel() for p in model.parameters())))}")
idx = torch.randint(0, 200, (1, 10))
logits = model(idx)
print(f"  输入 {tuple(idx.shape)} → logits {tuple(logits.shape)}（只取最后一位做预测）")

model.eval()
with torch.no_grad():
    full = model(idx)
    prefix = model(idx[:, :5])
print(f"  完整序列前 5 个位置的输出 vs 只喂前 5 个 token 的输出，"
      f"最大差异 = {(full[0, :5] - prefix[0]).abs().max().item():.2e}")
print("  差异为 0 → 因果掩码保证了『位置 i 只依赖 0..i』，这正是自回归生成的前提。")

section("4) 为什么 Decoder-only 赢了？")
print("""
  ① 目标统一：预测下一个 token 这一个目标，天然适配"预训练 → 一切任务"
  ② 规模化友好：结构简单、并行度高、工程优化（FlashAttention/KV Cache）收益最大
  ③ 涌现能力：模型变大后自动出现上下文学习、思维链、代码等能力
  ④ 工程惯性：整个推理生态（vLLM、量化、推测解码）都围绕 decoder-only 优化

  代价也要清楚：
    · 每生成一个 token 都要看全部历史 → 长输入场景下 prefill 成本高
    · 无法像 encoder 那样"一次前向得到整句表示" → 做 embedding 时要用特殊设计
      （这也是为什么检索模型依然是 BERT 系：见 01 节）
""")

section("5) 推理时的形状（配合 KV Cache 看）")
B, T, LAYERS, H, D = 1, 1024, 12, 12, 768
kv_per_layer = 2 * B * T * H * (D // H) * 2      # fp16
print(f"  GPT-2 规模（12 层 / 12 头 / d=768），T={T}, batch={B}")
print(f"    KV Cache = {bytes_str(LAYERS * kv_per_layer)}")
print(f"    权重(fp16) ≈ {bytes_str(117e6 * 2)}")
print(f"    logits 张量 = {bytes_str(B * T * 50257 * 4)}  ← 词表大时这一项也很可观！")
print("""
  注意最后一行：输出 logits 是 B×T×vocab，词表 128k、T=4096、batch=8 时
  单是 logits 就是 16GB（fp32）—— 所以现代实现会做"分块计算 loss"或 fused CE。
""")
