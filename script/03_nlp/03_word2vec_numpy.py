"""
03 · Word2Vec（Skip-gram + 负采样）纯 numpy 实现
==================================================
"分布式语义假设"：一个词的含义由它周围的词决定（You shall know a word by the company it keeps）。

Skip-gram：用中心词预测上下文词
    目标：最大化 P(context | center)，用负采样近似：
    loss = −log σ(v_c·u_o) − Σ_{k=1..K} log σ(−v_c·u_{n_k})

本脚本：
  1. 用"带语义结构的模板"造语料（动物一类、人物一类，人物内部还有性别/王权两个维度）
  2. 手推梯度并训练（负采样，向量化更新）
  3. 验证经典现象：语义相近的词聚在一起、king − man + woman ≈ queen
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import section, set_seed, subsection, timer

set_seed(5)

# ----------------------------------------------------------------------------------
# 造语料：模板随机采样，保证"同类词出现在相似的上下文里"
# ----------------------------------------------------------------------------------
rd = np.random.default_rng(1)
PERSON = {                                   # 人物: (代词, 住所, 饰品, 伴侣)
    "king":  ("he",  "palace", "crown", "queen"),
    "queen": ("she", "palace", "crown", "king"),
    "man":   ("he",  "house",  "hat",   "woman"),
    "woman": ("she", "house",  "hat",   "man"),
}
ANIMALS = ["cat", "dog", "mouse", "bird", "fish"]
FOODS = ["fish", "cheese", "seed", "worm", "bone"]
SPOTS = ["bed", "floor", "ground", "garden"]


def gen_sentence():
    if rd.random() < 0.5:
        p = PERSON[rd.choice(list(PERSON))]
        pron, place, item, partner = p[0], p[1], p[2], p[3]
        name = rd.choice(list(PERSON))
        _, p2, i2, part2 = PERSON[name]
        tpl = rd.integers(0, 6)
        if tpl == 0:
            return f"{name} rules the kingdom"
        if tpl == 1:
            return f"{name} lives in the {p2}"
        if tpl == 2:
            return f"{name} wears a {i2}"
        if tpl == 3:
            return f"{pron} loves {part2}"
        if tpl == 4:
            return f"{name} eats the {rd.choice(FOODS)}"
        return f"{name} feeds the {rd.choice(ANIMALS)}"
    a = rd.choice(ANIMALS)
    a2 = rd.choice(ANIMALS)
    tpl = rd.integers(0, 4)
    if tpl == 0:
        return f"{a} chases the {a2}"
    if tpl == 1:
        return f"{a} eats the {rd.choice(FOODS)}"
    if tpl == 2:
        return f"{a} sleeps on the {rd.choice(SPOTS)}"
    return f"{a} runs in the {rd.choice(SPOTS)}"


corpus = [gen_sentence().split() for _ in range(700)]
vocab = sorted({w for s in corpus for w in s})
w2i = {w: i for i, w in enumerate(vocab)}
i2w = {i: w for w, i in w2i.items()}
V = len(vocab)
print(f"语料 {len(corpus)} 句，词表 V={V}")
print(f"示例句子: {' | '.join(' '.join(s) for s in corpus[:4])}")

# ----------------------------------------------------------------------------------
# 生成 (center, context) 训练对
# ----------------------------------------------------------------------------------
WINDOW = 2
pairs = []
for sent in corpus:
    idx = [w2i[w] for w in sent]
    for i, c in enumerate(idx):
        for j in range(max(0, i - WINDOW), min(len(idx), i + WINDOW + 1)):
            if i != j:
                pairs.append((c, idx[j]))
print(f"训练对数量 = {len(pairs)}")

DIM = 32
rng = np.random.default_rng(0)
W_in = rng.normal(0, 0.1, (V, DIM))      # 中心词向量（最终用的词向量）
W_out = rng.normal(0, 0.1, (V, DIM))     # 上下文词向量（训练用）

counts = np.zeros(V)
for s in corpus:
    for w in s:
        counts[w2i[w]] += 1
probs = counts ** 0.75                   # 负采样分布：词频的 3/4 次方
probs /= probs.sum()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))


section("1) 训练 Skip-gram + 负采样")
EPOCHS, NEG = 60, 8
with timer("训练总耗时"):
    for ep in range(EPOCHS):
        LR = 0.15 * (1 - ep / EPOCHS) + 0.01          # 学习率线性衰减
        rng.shuffle(pairs)
        total_loss = 0.0
        for center, context in pairs:
            vc = W_in[center]                                       # (D,)
            # ---- 正样本 ----
            uo = W_out[context]
            g_pos = sigmoid(float(vc @ uo)) - 1.0                   # dL/d(v·u)
            grad_vc = g_pos * uo
            W_out[context] -= LR * g_pos * vc
            # ---- 负样本（一次性向量化，避免 Python 级循环）----
            negs = rng.choice(V, size=NEG, p=probs)
            Un = W_out[negs]                                        # (K, D)
            g_neg = sigmoid(Un @ vc)                                # (K,)
            grad_vc += g_neg @ Un
            np.add.at(W_out, negs, -LR * g_neg[:, None] * vc)
            W_in[center] -= LR * grad_vc
            total_loss += -np.log(sigmoid(float(vc @ uo)) + 1e-9) \
                          - np.log(sigmoid(-(Un @ vc)) + 1e-9).sum()
        if ep % 15 == 0 or ep == EPOCHS - 1:
            print(f"  epoch {ep:>3}  lr={LR:.3f}  loss={total_loss / len(pairs):.4f}")
print("  注：负采样的 loss 不保证单调下降（它是随机近似，且向量范数在增长）。")
print("      评估词向量的标准做法是下面两节的『最近邻 / 类比』，而不是看 loss。")


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def most_similar(word, topn=5):
    if word not in w2i:
        return []
    v = W_in[w2i[word]]
    sims = [(cosine(v, W_in[i]), i2w[i]) for i in range(V) if i != w2i[word]]
    sims.sort(reverse=True)
    return sims[:topn]


section("2) 最近邻：语义相近的词聚在一起了吗？")
for w in ["cat", "dog", "king", "queen", "fish", "eats"]:
    nn = most_similar(w)
    print(f"  {w:<6} -> " + ", ".join(f"{n}({s:.2f})" for s, n in nn))
print("\n  预期：cat/dog/mouse/bird/fish 互相靠近；king/queen/man/woman 互相靠近。")

section("3) 词向量类比：king − man + woman ≈ ?")


def analogy(a, b, c, topn=3):
    """a : b = c : ?   →  v(?) ≈ v(b) − v(a) + v(c)"""
    q = W_in[w2i[b]] - W_in[w2i[a]] + W_in[w2i[c]]
    sims = [(cosine(q, W_in[i]), i2w[i]) for i in range(V) if i not in (w2i[a], w2i[b], w2i[c])]
    sims.sort(reverse=True)
    return sims[:topn]


print(f"  king − man + woman -> {[(w, round(s, 3)) for s, w in analogy('king', 'man', 'woman')]}")
print(f"  queen − woman + man -> {[(w, round(s, 3)) for s, w in analogy('queen', 'woman', 'man')]}")
print(f"  cat − dog + fish -> {[(w, round(s, 3)) for s, w in analogy('cat', 'dog', 'fish')]}")
print("  向量加减真的对应语义维度：『性别轴』和『王权轴』被学成了两个近似正交的方向。")

section("4) 为什么需要负采样？")
print(f"""
  · 原始 softmax 的分母要对全部 V={V} 个词求和 → 每个样本 O(V)，大词表（128k）下不可接受
  · 负采样把"V 分类"变成"K+1 个二分类" → 每步 O(K)，K 通常取 5~20
  · 现代 LLM 依然要算 full softmax（因为要输出精确概率），但会用
    分块计算 / fused cross-entropy（第 7 章）来降低显存与访存开销
""")

section("5) 静态词向量 → 上下文词向量")
print("""
  Word2Vec 的问题：'bank'（银行 / 河岸）只有一个向量，无法消歧
  ELMo / BERT / LLM 的解法：词向量由整个句子动态计算 → 同一个词在不同上下文里向量不同
  这条演进路线就是下一章 Transformer 的动机之一。
""")
