"""
07 · 训练一个"反转序列"模型，然后看它的注意力到底在关注什么
==============================================================
为什么选"反转"任务？
  因为正确答案的注意力模式是已知的：输出第 i 个位置时，应该去看输入的第 (T−1−i) 个位置
  → 注意力图应该呈现出清晰的反对角线。
如果模型真学会了这个任务，我们就应该能在注意力图里"看见"它的解法。

这就是可解释性研究的基本套路：找一个已知解法的任务 → 训练 → 反查模型内部的表示。
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import ascii_heatmap, has_matplotlib, require_torch, save_fig, section, set_seed, subsection, timer

set_seed(8)
torch = require_torch("07_attention_visualization")
nn = torch.nn
F = torch.nn.functional

VOCAB, D_MODEL, H, LAYERS, MAX_LEN = 20, 64, 2, 2, 64
BOS, SEP = 20, 21


def make_batch(batch_size, seq_len):
    """x = [BOS, a₁..aₖ, SEP, aₖ..a₁]   y 只在 SEP 之后计 loss"""
    toks = torch.randint(0, VOCAB - 2, (batch_size, seq_len))
    bos = torch.full((batch_size, 1), BOS)
    sep = torch.full((batch_size, 1), SEP)
    rev = torch.flip(toks, dims=[1])
    x = torch.cat([bos, toks, sep, rev], dim=1)
    y = x.clone()
    y[:, :seq_len + 2] = -100
    return x, y


class ReverseTransformer(nn.Module):
    """手写两层的 decoder-only Transformer，方便把注意力权重取出来。"""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB + 2, D_MODEL)
        self.pos = nn.Embedding(MAX_LEN, D_MODEL)
        self.qkv = nn.ModuleList([nn.Linear(D_MODEL, 3 * D_MODEL, bias=False) for _ in range(LAYERS)])
        self.proj = nn.ModuleList([nn.Linear(D_MODEL, D_MODEL, bias=False) for _ in range(LAYERS)])
        self.ff = nn.ModuleList([nn.Sequential(
            nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(), nn.Linear(4 * D_MODEL, D_MODEL))
            for _ in range(LAYERS)])
        self.ln1 = nn.ModuleList([nn.LayerNorm(D_MODEL) for _ in range(LAYERS)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(D_MODEL) for _ in range(LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB + 2, bias=False)
        self.h, self.dh = H, D_MODEL // H

    def forward(self, idx, return_attn=False):
        B, T = idx.shape
        x = self.emb(idx) + self.pos(torch.arange(T, device=idx.device)[None])
        causal = torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
        attns = []
        for l in range(LAYERS):
            h = self.ln1[l](x)
            qkv = self.qkv[l](h).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]                       # (B,h,T,dh)
            scores = (q @ k.transpose(-2, -1)) / (self.dh ** 0.5)
            scores = scores.masked_fill(~causal, float("-inf"))
            w = torch.softmax(scores, dim=-1)
            attns.append(w.detach())
            ctx = (w @ v).transpose(1, 2).reshape(B, T, D_MODEL)
            x = x + self.proj[l](ctx)
            x = x + self.ff[l](self.ln2[l](x))
        logits = self.head(self.ln_f(x))
        return (logits, attns) if return_attn else logits


section("1) 训练反转任务（k=6，1200 步）")
model = ReverseTransformer()
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
print(f"  参数量 = {sum(p.numel() for p in model.parameters()):,}")

with timer("训练耗时"):
    for step in range(1200):
        x, y = make_batch(64, 6)
        logits = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB + 2), y[:, 1:].reshape(-1),
                               ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 300 == 0 or step == 1199:
            with torch.no_grad():
                pred = logits[:, :-1].argmax(-1)
                m = (y[:, 1:] != -100)
                acc = ((pred == y[:, 1:]) & m).sum().item() / m.sum().item()
            print(f"  step {step:>4}  loss={loss.item():.4f}  token 准确率={acc:.4f}")

section("2) 看注意力图：模型『在看』哪里？")
model.eval()
with torch.no_grad():
    toks = torch.randint(0, VOCAB - 2, (1, 6))
    x = torch.cat([torch.tensor([[BOS]]), toks, torch.tensor([[SEP]])], dim=1)
    full = torch.cat([x, torch.flip(toks, dims=[1])], dim=1)
    _, attns = model(full, return_attn=True)
print(f"  输入序列: {full[0].tolist()}")
print("  （BOS, a₁..a₆, SEP, 然后是期望输出的反转序列）")

for l in range(LAYERS):
    for head in range(H):
        w = attns[l][0, head].numpy()
        # 只看"生成阶段"的 query（SEP 之后的 6 个位置）
        gen_rows = w[8:14]
        print(f"\n  第 {l} 层 · 头 {head}（行=生成位置，列=被关注的位置 0..13）：")
        print(ascii_heatmap(gen_rows, width=56, height=8))
        for r in range(gen_rows.shape[0]):
            print(f"    生成第 {r} 个 token 时最关注位置 {int(np.argmax(gen_rows[r]))} "
                  f"（权重 {gen_rows[r].max():.2f}）")

print("""
  期望模式：生成第 r 个 token 时，应该关注输入区间的第 (5 − r) 个 token（即位置 6−r）。
  如果看到清晰的反对角线，说明模型学到的正是"反转"这个算法，而不是死记硬背。
""")

section("3) 真实 LLM 里被发现的同类模式")
print("""
  · Induction Head（归纳头）: 看到 "A B ... A" 时，去关注前一个 A 后面的 B —— 上下文学习的基石
  · Previous Token Head      : 基本只看前一个 token（负责局部语法）
  · Duplicate Token Head     : 关注与当前 token 相同的历史 token（复制/去重）
  · Attention Sink           : 大量注意力集中在第一个 token（BOS）上，
    所以很多推理框架会保留这个"锚点"，丢弃它会导致模型崩掉
""")

if has_matplotlib():
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(LAYERS, H, figsize=(3.2 * H, 3.2 * LAYERS))
    for l in range(LAYERS):
        for head in range(H):
            ax = axes[l, head] if LAYERS > 1 else axes[head]
            ax.imshow(attns[l][0, head].numpy(), cmap="viridis")
            ax.set_title(f"layer {l} head {head}")
            ax.set_xlabel("key")
            ax.set_ylabel("query")
    plt.tight_layout()
    save_fig(fig, Path(__file__).parent / "outputs" / "07_attention_maps.png")

section("4) 小结：可视化能给你什么")
print("""
  ① 调试：注意力全均匀 / 全集中在某一列，通常意味着训练有问题
  ② 剪枝与加速：某些头几乎没用 → 可以砍掉（head pruning）
  ③ 可解释性：把"模型在想什么"变成可观察的矩阵，是机制可解释性的起点
  局限：注意力权重 ≠ 贡献度（后面的 FFN 可能把它放大或忽略），
        更严谨的做法是 activation patching / attribution，别过度解读注意力图。
""")
