"""
05 · 真正训练一个 Transformer：复制任务 + LLM 训练技巧
======================================================
任务（Copy）：输入 [BOS, a₁..aₖ, <copy>, a₁..aₖ]，模型要在 <copy> 之后把这段原样再输出一遍
  只有后半段的预测算 loss（前半段是"读"，后半段是"写"）

借这个小任务，把 LLM 预训练会用到的工程技巧全过一遍：
  · Teacher Forcing（用真值做上文，而不是自己生成的）
  · 学习率 warmup + 余弦衰减（LLM 训练的标配）
  · 梯度裁剪（防止 loss 尖峰把训练带崩）
  · 长度外推测试（训练 k=8，测试 k=24 会怎样）

⚠ 工程踩坑：本脚本刻意不用 nn.TransformerEncoderLayer —— 它在 eval 模式下会走
  PyTorch 的 fast path，与自定义 attn_mask 组合时容易踩坑。自己写注意力反而更安全、
  也更便于理解。第 7 章你会知道 fast path 背后其实是融合算子。
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import human_num, require_torch, section, set_seed, subsection, timer

set_seed(7)
torch = require_torch("05_train_toy_copy")
nn = torch.nn
F = torch.nn.functional

VOCAB = 24                     # 0..20 普通 token，21=<copy>, 22=<bos>, 23=<pad>
COPY_TOK = 21
BOS, PAD = 22, 23
D_MODEL, H, LAYERS, MAX_LEN = 64, 4, 2, 128


def make_batch(batch_size, seq_len):
    """构造 [BOS, a₁..aₖ, <copy>, a₁..aₖ]，标签只在后半段有效。"""
    toks = torch.randint(0, 20, (batch_size, seq_len))
    bos = torch.full((batch_size, 1), BOS)
    cp = torch.full((batch_size, 1), COPY_TOK)
    x = torch.cat([bos, toks, cp, toks], dim=1)              # (B, 2k+2)
    y = x.clone()
    y[:, :seq_len + 2] = -100                                # 前半段不计 loss（ignore_index）
    return x, y


class Block(nn.Module):
    """手写的一个 decoder block：Pre-LN + 因果自注意力 + FFN。"""

    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        self.h, self.dh = num_heads, d_model // num_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(4 * d_model, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, causal_mask):
        B, T, D = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) / (self.dh ** 0.5)
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        w = torch.softmax(scores, dim=-1)
        ctx = (self.drop(w) @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(ctx)
        x = x + self.ff(self.ln2(x))
        return x


class TinyTransformerLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Embedding(MAX_LEN, D_MODEL)
        self.blocks = nn.ModuleList([Block(D_MODEL, H) for _ in range(LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        x = self.emb(idx) + self.pos(torch.arange(T, device=idx.device)[None])
        causal = torch.ones(T, T, dtype=torch.bool, device=idx.device).tril()
        for blk in self.blocks:
            x = blk(x, causal)
        return self.head(self.ln_f(x))


def evaluate(model, k, batch=64):
    model.eval()
    with torch.no_grad():
        x, y = make_batch(batch, k)
        logits = model(x)
        pred = logits[:, :-1].argmax(-1)
        m = (y[:, 1:] != -100)
        return ((pred == y[:, 1:]) & m).sum().item() / m.sum().item()


section("1) 训练（序列长度 k=8，batch=64，800 步）")
model = TinyTransformerLM()
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
steps, warmup, total = 800, 40, 800
print(f"  参数量 = {human_num(float(sum(p.numel() for p in model.parameters())))}")


def lr_lambda(step):
    if step < warmup:
        return step / max(warmup, 1)                          # warmup：从 0 线性升到峰值
    prog = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + np.cos(np.pi * prog))                   # 余弦衰减到 0


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

with timer("训练耗时"):
    for step in range(steps):
        model.train()
        x, y = make_batch(64, 8)
        logits = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), y[:, 1:].reshape(-1),
                               ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 梯度裁剪
        opt.step()
        sched.step()
        if step % 100 == 0 or step == steps - 1:
            print(f"  step {step:>4}  loss={loss.item():.4f}  "
                  f"token 准确率={evaluate(model, 8):.4f}  lr={sched.get_last_lr()[0]:.2e}")

section("2) 长度外推：训练 k=8，测试更长会怎样")
for k in [4, 8, 12, 16, 24]:
    print(f"  k={k:<3}  token 准确率={evaluate(model, k):.4f}")
print("""
  观察：训练长度附近表现好，超出后明显下降 —— 这就是"长度外推"问题。
  本模型用的是【可学习的绝对位置编码】，它压根没见过 k>8 时的那些位置向量。
  改进方向：换成 RoPE / ALiBi（见 03 节），或者做长上下文继续训练。
""")

section("3) 自回归生成：一步步把复制结果『写』出来")
model.eval()
with torch.no_grad():
    k = 8                                   # 用训练时见过的长度（换成长度外推会明显变差）
    toks = torch.randint(0, 20, (1, k))
    ctx = torch.cat([torch.tensor([[BOS]]), toks, torch.tensor([[COPY_TOK]])], dim=1)
    print(f"  待复制: {toks.tolist()[0]}")
    generated = []
    for _ in range(k):
        logits = model(ctx)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        generated.append(int(nxt.item()))
        ctx = torch.cat([ctx, nxt], dim=1)
    print(f"  模型输出: {generated}")
    print(f"  是否完全一致: {generated == toks.tolist()[0]}")
print("""
  注意这里用的是"自回归生成"：每一步把自己的输出拼回输入（推理时没有 teacher forcing）。
  这也是 LLM 聊天的真实工作方式 —— 一次前向只出一个 token。
""")

section("4) 训练技巧小结（直接迁移到 LLM 预训练）")
print("""
  ① Teacher Forcing：训练时喂真值，推理时喂自己的输出 → 存在 exposure bias，
     大模型的缓解手段是 RLHF 阶段的在线采样
  ② Warmup：训练最初几千步用小 lr，避免 Adam 二阶动量未稳定时把模型带崩
  ③ 余弦衰减：把 lr 降到接近 0，通常能把最终 loss 再降一截
  ④ 梯度裁剪：LLM 训练出现 loss spike 时的保命手段（典型 max_norm=1.0）
  ⑤ 混合精度：bf16/fp16 计算 + fp32 主权重（第 7 章会讲算子层面的原因）
""")
