"""
01 · Encoder-only 代表：BERT（双向注意力 + 掩码语言模型）
==========================================================
BERT 的两个关键词：
  ① 双向注意力：没有因果掩码，每个位置都能看到整句 → 强大的"理解"能力
  ② MLM 预训练：随机盖住 15% 的 token，让模型去预测被盖住的词

代价：BERT 不能自然地做生成（没有自回归结构），所以今天的聊天 LLM 全是 decoder-only。
但 BERT 系模型依然是【检索 / 分类 / 序列标注 / embedding】场景的主力（体积小、速度快）。

本脚本：手写 BERT 结构并在玩具语料上跑一次真正的 MLM 训练，
最后演示"同一个词在不同句子里向量不同" —— 这正是 BERT 相比 Word2Vec 的飞跃。
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import ascii_heatmap, require_torch, section, set_seed, subsection, timer

set_seed(3)
torch = require_torch("01_encoder_only_bert")
nn = torch.nn
F = torch.nn.functional

# ----------------------------------------------------------------------------------
# 玩具语料：有明确的句法/语义结构
# ----------------------------------------------------------------------------------
CORPUS = [
    "the cat sits on the mat", "the dog sits on the floor", "a cat eats the fish",
    "a dog eats the bone", "the fish swims in water", "the bird flies in sky",
    "my cat sleeps on bed", "my dog sleeps on floor", "the cat catches the mouse",
    "the dog catches the ball", "i like cats and dogs", "cats and dogs are pets",
]
words = sorted({w for s in CORPUS for w in s.split()})
VOCAB = ["[PAD]", "[CLS]", "[SEP]", "[MASK]"] + words
w2i = {w: i for i, w in enumerate(VOCAB)}
i2w = {i: w for w, i in w2i.items()}
V = len(VOCAB)
print(f"词表大小 V={V}")


def encode(sent):
    return [w2i["[CLS]"]] + [w2i[w] for w in sent.split()] + [w2i["[SEP]"]]


data = [encode(s) for s in CORPUS]
MAXLEN = max(len(d) for d in data)


def make_mlm_batch(batch_size=16, mask_prob=0.15):
    """随机选句子，随机遮盖 15% 的 token（简化：80% 换成 [MASK]，20% 保持不变）"""
    idxs = np.random.randint(0, len(data), batch_size)
    x = np.zeros((batch_size, MAXLEN), dtype=np.int64)
    y = np.full((batch_size, MAXLEN), -100, dtype=np.int64)
    for r, i in enumerate(idxs):
        seq = data[i].copy()
        # padding 到 MAXLEN
        seq = seq + [0] * (MAXLEN - len(seq))
        for j in range(1, len(data[i]) - 1):              # 不遮盖 [CLS] / [SEP]
            if np.random.rand() < mask_prob:
                y[r, j] = seq[j]
                seq[j] = w2i["[MASK]"] if np.random.rand() < 0.8 else seq[j]
        x[r] = seq
    return torch.tensor(x), torch.tensor(y)


# ----------------------------------------------------------------------------------
class BertBlock(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True, dropout=0.0)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                nn.Linear(4 * d_model, d_model))

    def forward(self, x, need_weights=False):
        h = self.ln1(x)
        if need_weights:
            o, w = self.attn(h, h, h, need_weights=True, average_attn_weights=False)
        else:
            o = self.attn(h, h, h)[0]
        x = x + o
        x = x + self.ff(self.ln2(x))
        return (x, w) if need_weights else x


class TinyBERT(nn.Module):
    def __init__(self, vocab, d_model=64, heads=4, layers=2, maxlen=64):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, d_model)
        self.type_emb = nn.Embedding(2, d_model)          # segment A/B（NSP 用）
        self.ln = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList([BertBlock(d_model, heads) for _ in range(layers)])
        self.mlm_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                      nn.LayerNorm(d_model), nn.Linear(d_model, vocab))

    def forward(self, ids, need_weights=False):
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device)[None]
        x = self.tok_emb(ids) + self.pos_emb(pos) + self.type_emb(torch.zeros(B, T, dtype=torch.long, device=ids.device))
        x = self.ln(x)
        w = None
        for blk in self.blocks:
            if need_weights:
                x, w = blk(x, need_weights=True)
            else:
                x = blk(x)
        return (self.mlm_head(x), w) if need_weights else self.mlm_head(x)


section("1) BERT 的三个 Embedding 相加")
model = TinyBERT(V, maxlen=MAXLEN + 4)
print(f"  参数量 = {sum(p.numel() for p in model.parameters()):,}")
print("""
  input = token_emb + position_emb + segment_emb
    · token    : 词本身
    · position : 位置（BERT 用的是可学习绝对位置）
    · segment  : 属于句子 A 还是句子 B（为 NSP 任务服务）
""")

section("2) MLM 训练（600 步，余弦衰减）")
STEPS = 600
opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
with timer("训练耗时"):
    for step in range(STEPS + 1):
        x, y = make_mlm_batch(16)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, V), y.view(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 150 == 0 or step == STEPS:
            with torch.no_grad():
                pred = logits.argmax(-1)
                m = (y != -100)
                acc = ((pred == y) & m).sum().item() / max(m.sum().item(), 1)
            print(f"  step {step:>4}  loss={loss.item():.4f}  MLM 准确率={acc:.4f}")

section("3) 填空演示：模型真的学会了语料里的规律")
model.eval()
tests = ["the cat sits on the [MASK]", "a dog eats the [MASK]", "the [MASK] flies in sky"]


def encode_with_special(sent):
    """演示时也要补上 [CLS] / [SEP]，否则输入分布与训练时不一致。"""
    return torch.tensor([[w2i["[CLS]"]] + [w2i.get(w, w2i["[MASK]"]) for w in sent.split()]
                         + [w2i["[SEP]"]]])


for t in tests:
    ids = encode_with_special(t)
    with torch.no_grad():
        logits = model(ids)
    pos = int((ids[0] == w2i["[MASK]"]).nonzero()[0])
    top = logits[0, pos].topk(4)
    print(f"  {t:<34} → " + ", ".join(f"{i2w[int(i)]}({v:.2f})" for v, i in zip(top.values, top.indices)))

section("4) BERT 的核心价值：上下文相关的词向量")
sents = ["the cat sits on the mat", "the dog catches the ball"]


def contextual_vectors(sent):
    ids = torch.tensor([[w2i.get(w, w2i["[MASK]"]) for w in sent.split()]])
    with torch.no_grad():
        logits, w = model(ids, need_weights=True)
    return logits[0], w[0]          # 用 MLM head 之前的隐藏状态近似"上下文向量"


# 取最后一层之前的隐藏状态：这里用 mlm_head 的输入做演示需要改一下，简单起见直接取 embedding 后的输出
def hidden_states(sent):
    ids = encode_with_special(sent)
    with torch.no_grad():
        x = model.tok_emb(ids) + model.pos_emb(torch.arange(ids.shape[1])[None])
        x = model.ln(x)
        for blk in model.blocks:
            x = blk(x)
    return x[0]


h1 = hidden_states(sents[0])
h2 = hidden_states(sents[1])
sim_same_pos = float(F.cosine_similarity(h1[1], h2[1], dim=0))       # 都是第一个实词
print(f"  '{sents[0]}' 与 '{sents[1]}' 的第 1 个实词向量余弦相似度 = {sim_same_pos:.3f}")
h3 = hidden_states("the cat catches the mouse")
sim_cat_cat = float(F.cosine_similarity(h1[1], h3[1], dim=0))
print(f"  两句都以 'the cat' 开头时，cat 的向量相似度 = {sim_cat_cat:.3f}")
print("""
  结论：同一个词在不同上下文里向量不同 → 这就是『上下文相关表示』，
  也是 BERT 之后所有 LLM 的基础（Word2Vec 那种静态向量已经不够用了）。
""")

section("5) 双向注意力自证：没有因果掩码")
_, w = model(torch.tensor([encode("the cat sits on the mat")]), need_weights=True)
w0 = w[0, 0].detach().numpy()          # 第 0 层第 0 头
print(ascii_heatmap(w0[:8, :8], width=40, height=10, title="  注意力（上三角不为 0 = 能看到右边）："))
print("  若这是 GPT，上三角必须全为 0。BERT 能看到整句，所以更擅长理解类任务。")

section("6) 现代视角：BERT 还值得学吗")
print("""
  值得，但要分清场景：
    · 检索 / RAG 的 embedding 模型（BGE、GTE、M3E）→ 都是 BERT 系（encoder-only）
    · 分类、NER、意图识别 → BERT 系仍是性价比之王（几百 MB，毫秒级）
    · 生成、对话、推理 → decoder-only LLM
  另外：LLM 时代的"判别式"需求（打分、排序、rerank）也多用 encoder 架构，
  因为一次前向就能得到整句的表示，不必自回归。
""")
