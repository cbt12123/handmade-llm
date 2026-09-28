"""
04 · 从零搭一个完整的 Transformer（Encoder + Decoder），纯 PyTorch
====================================================================
一个 Transformer Block = 注意力子层 + FFN 子层，每个子层都套：

        x → Sublayer(x) → x + Dropout(Sublayer(LayerNorm(x)))     后残差 + Pre-LN

本脚本把每个组件都写成一个 nn.Module，最后拼出 Encoder / Decoder，
并验证输出形状、参数量、以及"因果掩码是否真的挡住了未来信息"。
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import human_num, require_torch, section, set_seed, subsection

set_seed(4)
torch = require_torch("04_transformer_block")
nn = torch.nn
F = torch.nn.functional


# ----------------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model, self.h, self.d_head = d_model, num_heads, d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_kv=None, mask=None, is_causal=False, need_weights=False):
        """x_q/x_kv: (B, T, d_model)；mask: (B, T_kv) 或 (T_q, T_kv) 的 bool，True 表示【可见】"""
        x_kv = x_q if x_kv is None else x_kv
        B, Tq, _ = x_q.shape
        Tk = x_kv.shape[1]
        q = self.W_q(x_q).view(B, Tq, self.h, self.d_head).transpose(1, 2)     # (B,h,Tq,dh)
        k = self.W_k(x_kv).view(B, Tk, self.h, self.d_head).transpose(1, 2)
        v = self.W_v(x_kv).view(B, Tk, self.h, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)              # (B,h,Tq,Tk)
        if is_causal:                                                          # 因果（下三角）掩码
            causal = torch.ones(Tq, Tk, dtype=torch.bool, device=x_q.device).tril()
            scores = scores.masked_fill(~causal, float("-inf"))
        if mask is not None:                                                   # padding 掩码
            m = mask[:, None, None, :] if mask.dim() == 2 else mask
            scores = scores.masked_fill(~m, float("-inf"))
        w = torch.softmax(scores, dim=-1)
        w = self.dropout(w)
        ctx = (w @ v).transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        return (self.W_o(ctx), w) if need_weights else self.W_o(ctx)


class FeedForward(nn.Module):
    """逐位置的 MLP：d_model → 4·d_model → d_model（原版设定）"""
    def __init__(self, d_model, d_ff=None, dropout=0.1, act="gelu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if act == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-LN 结构（现代 LLM 通用）：先 LayerNorm 再进子层，残差直连。"""
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1, cross_attn=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.cross = None
        if cross_attn:
            self.ln_cross = nn.LayerNorm(d_model)
            self.cross = MultiHeadAttention(d_model, num_heads, dropout)

    def forward(self, x, memory=None, mask=None, is_causal=False):
        x = x + self.attn(self.ln1(x), mask=mask, is_causal=is_causal)         # 自注意力
        if self.cross is not None and memory is not None:
            x = x + self.cross(self.ln_cross(x), x_kv=memory)                  # 交叉注意力
        x = x + self.ff(self.ln2(x))                                           # FFN
        return x


# ----------------------------------------------------------------------------------
section("1) 组装一个 4 层 Transformer")
D_MODEL, H, LAYERS = 64, 4, 4
enc_layer = TransformerBlock(D_MODEL, H)
dec_layer = TransformerBlock(D_MODEL, H, cross_attn=True)
print(f"  单层 Encoder 参数量 = {human_num(float(sum(p.numel() for p in enc_layer.parameters())))}")
print(f"  单层 Decoder 参数量 = {human_num(float(sum(p.numel() for p in dec_layer.parameters())))}（多了交叉注意力）")

B, T = 2, 12
x = torch.randn(B, T, D_MODEL)
out = enc_layer(x)
print(f"\n  输入 {tuple(x.shape)} → 输出 {tuple(out.shape)} （形状不变，这是能堆叠 N 层的前提）")

section("2) 参数量拆解（d_model=64, h=4）")
attn_p = 4 * D_MODEL * D_MODEL
ff_p = 2 * D_MODEL * (4 * D_MODEL)
print(f"  注意力部分 W_qkv + W_o = {attn_p}")
print(f"  FFN 部分 两个 Linear  = {ff_p}   ← FFN 占大头（约 2/3）")
print(f"  LayerNorm 等          = {sum(p.numel() for p in enc_layer.parameters()) - attn_p - ff_p}")
print("  规律：FFN 的参数量 = 2·d·4d = 8d²，注意力 = 4d² → 总约 12d²/层。")
print("  估算 LLM 参数：L 层 × 12d² ≈ 参数量（Llama-7B: 32 层 × 12 × 4096² ≈ 6.4B ✔）")

section("3) 因果掩码：验证未来信息真的被挡住了")
layer = TransformerBlock(D_MODEL, H)
layer.eval()                        # ← 必须关掉 dropout：否则两次前向的随机掩码不同，验证会"假失败"
torch.manual_seed(0)
x1 = torch.randn(1, 6, D_MODEL)
out_full = layer(x1, is_causal=True)
# 只喂前 3 个 token（同样的权重），第 3 个位置的输出应该完全一致
out_prefix = layer(x1[:, :3], is_causal=True)
max_diff = (out_full[0, :3] - out_prefix[0]).abs().max().item()
print(f"  完整序列前 3 个位置 vs 只喂前 3 个 token 的输出，最大差异 = {max_diff:.2e}")
print("  差异为 0 → 说明因果掩码生效：位置 i 的输出只依赖 0..i（这就是 GPT 类自回归的基础）")

section("4) 掩码自注意力的注意力图长什么样")
attn = MultiHeadAttention(D_MODEL, H)
torch.manual_seed(1)
_, w = attn(torch.randn(1, 8, D_MODEL), is_causal=True, need_weights=True)
w0 = w[0, 0].detach().numpy()                       # 第 0 个头
print("  （行=query 位置，列=key 位置；上三角应全为 0）")
for i in range(8):
    print("   " + " ".join(f"{v:.2f}" for v in w0[i]))

section("5) 完整 Encoder-Decoder 前向（翻译模型的骨架）")
class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.enc_layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, dropout=dropout) for _ in range(num_layers)])
        self.dec_layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, dropout=dropout, cross_attn=True)
             for _ in range(num_layers)])

    def forward(self, src, tgt, src_mask=None):
        memory = src
        for blk in self.enc_layers:
            memory = blk(memory, mask=src_mask)
        y = tgt
        for blk in self.dec_layers:
            y = blk(y, memory=memory, is_causal=True)
        return y, memory


model = Transformer(D_MODEL, H, num_layers=2)
src = torch.randn(2, 10, D_MODEL)
tgt = torch.randn(2, 7, D_MODEL)
y, mem = model(src, tgt)
print(f"  src {tuple(src.shape)} → memory {tuple(mem.shape)}")
print(f"  tgt {tuple(tgt.shape)} → decoder 输出 {tuple(y.shape)}")
print(f"  总参数量 = {human_num(float(sum(p.numel() for p in model.parameters())))}")
print("  再加一个 nn.Linear(d_model, vocab_size) 就是完整的翻译/生成模型。")

section("6) 原版 vs 现代实现的 5 个差异（重要）")
print("""
  ① LayerNorm 位置    : 原版 Post-LN（残差之后）→ 现代 Pre-LN（残差之前），训练更稳
  ② 归一化            : LayerNorm → RMSNorm（Llama），少算均值和偏置，更快
  ③ 激活函数          : ReLU → GELU → SwiGLU（Llama / PaLM）
  ④ 位置编码          : 正弦/可学习 → RoPE / ALiBi
  ⑤ 注意力实现        : 朴素 softmax(QKᵀ)V → FlashAttention（不落地 T×T 矩阵）
  这些改动单个都不大，但合起来决定了"能不能训得动 100B 模型"。
""")
