"""
02 · n-gram 语言模型 + 困惑度：LLM 的"史前版本"
================================================
语言模型的定义：给一段文本一个概率 P(w₁, w₂, ..., wₙ)

n-gram 假设（马尔可夫假设）：每个词只依赖前 n−1 个词
    P(wᵢ | w₁..wᵢ₋₁) ≈ P(wᵢ | wᵢ₋ₙ₊₁ .. wᵢ₋₁)

本脚本：
  1. 手写 bigram / trigram 的 MLE 估计
  2. 三大平滑方法：add-k、插值(interpolation)、回退(backoff/Katz 简化版)
  3. 用困惑度 perplexity 评价模型 —— 这个指标一直沿用到了今天的 LLM
  4. 用模型随机生成句子（体验"概率就是一切"）
"""
import sys
from pathlib import Path
from collections import defaultdict
import math
import random

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common_utils import section, set_seed, subsection

set_seed(3)

CORPUS = [
    "i like deep learning",
    "i like machine learning",
    "i love reading books about machine learning",
    "deep learning needs data",
    "machine learning needs data and compute",
    "i love reading books",
    "books about deep learning are fun",
    "compute makes machine learning possible",
]
TOKENS = []
for line in CORPUS:
    TOKENS.append(["<s>"] * 1 + line.split() + ["</s>"])
FLAT = [t for s in TOKENS for t in s]
VOCAB = sorted(set(FLAT))
V = len(VOCAB)
print(f"语料：{len(CORPUS)} 句，{len(FLAT)} 个 token，词表大小 V={V}")

# ----------------------------------------------------------------------------------
section("1) 统计 n-gram 计数")
unigram = defaultdict(int)
bigram = defaultdict(int)
trigram = defaultdict(int)
for sent in TOKENS:
    for i, w in enumerate(sent):
        unigram[w] += 1
        if i >= 1:
            bigram[(sent[i - 1], w)] += 1
        if i >= 2:
            trigram[(sent[i - 2], sent[i - 1], w)] += 1
print(f"  unigram 类型数={len(unigram)}  bigram 类型数={len(bigram)}  trigram 类型数={len(trigram)}")
print(f"  例：count('learning')={unigram['learning']}, count('machine','learning')={bigram[('machine','learning')]}")

section("2) 无平滑的 MLE 与它的致命问题")
def p_mle_bigram(w2, w1):
    return bigram[(w1, w2)] / unigram[w1] if unigram[w1] else 0.0


print(f"  P(learning | machine) = {p_mle_bigram('learning', 'machine'):.3f}")
print(f"  P(books | machine)    = {p_mle_bigram('books', 'machine'):.3f}   ← 0！")
print("  只要语料里没出现过这个组合，概率就是 0；连乘后整句概率变 0 → 模型完全不可用。")

section("3) add-k（拉普拉斯）平滑")
def p_addk(w2, w1, k=0.5):
    return (bigram[(w1, w2)] + k) / (unigram[w1] + k * V)


for k in [0.1, 0.5, 1.0]:
    p_known = p_addk("learning", "machine", k)
    p_unknown = p_addk("books", "machine", k)
    print(f"  k={k:<4} P(learning|machine)={p_known:.4f}   P(books|machine)={p_unknown:.4f}")

section("4) 线性插值平滑（把不同阶的 n-gram 混起来）")
def p_interp(w3, w2, w1, l1=0.2, l2=0.3, l3=0.5):
    p3 = trigram[(w1, w2, w3)] / bigram[(w1, w2)] if bigram[(w1, w2)] else 0.0
    p2 = bigram[(w2, w3)] / unigram[w2] if unigram[w2] else 0.0
    p1 = unigram[w3] / len(FLAT)
    return l3 * p3 + l2 * p2 + l1 * p1


print(f"  P(learning | i like)     = {p_interp('learning', 'like', 'i'):.4f}")
print(f"  P(dog | i like)          = {p_interp('dog', 'like', 'i'):.4f}   ← 不再是 0")
print("  插值平滑的思想（用低阶统计兜底）在今天的 Kneser-Ney 平滑里依然是核心。")

section("5) 困惑度 Perplexity = exp(−1/N · Σ log P(wᵢ|context))")
def perplexity(sent, smooth_fn, n=2):
    sent = ["<s>"] + list(sent) + ["</s>"]
    logp = 0.0
    cnt = 0
    for i in range(1, len(sent)):
        ctx = sent[i - 1] if n == 2 else (sent[i - 2] if i >= 2 else "<s>")
        p = smooth_fn(sent[i], ctx)
        p = max(p, 1e-10)
        logp += math.log(p)
        cnt += 1
    return math.exp(-logp / cnt)


test = "i like deep learning"
print(f"  测试句: {test}")
print(f"  bigram + add-k(0.5) 的 perplexity = {perplexity(test.split(), lambda w2, w1: p_addk(w2, w1, 0.5)):.2f}")
print(f"  bigram 无平滑     的 perplexity = {perplexity(test.split(), lambda w2, w1: p_mle_bigram(w2, w1)):.2f}")
print("  perplexity 越低越好；数值意义 ≈『模型在每一步平均有多困惑（相当于在几个词之间犹豫）』")

section("6) 用 bigram 模型生成句子")
rng = random.Random(0)


def generate(start="<s>", max_len=10):
    out = []
    cur = start
    for _ in range(max_len):
        cands = [(w2, c) for (w1, w2), c in bigram.items() if w1 == cur]
        if not cands:
            break
        words, weights = zip(*[(w, c) for w, c in cands])
        nxt = rng.choices(words, weights=weights)[0]
        if nxt == "</s>":
            break
        out.append(nxt)
        cur = nxt
    return " ".join(out)


for _ in range(5):
    print("   ->", generate())
print("  生成的句子语法破碎但词汇连贯 —— 正是 n-gram 的典型特征（只能看到 1 个词的窗口）。")

section("7) 从 n-gram 到神经网络语言模型")
print("""
  n-gram 的根本缺陷：窗口只有 n−1 个词，且参数随 n 指数爆炸（V^n）
  神经网络 LM 的解法：把词映射成稠密向量（embedding），用模型函数拟合 P(wᵢ | 前文)
    → 参数与词表线性相关，且能泛化到没见过的组合（"猫/狗" 共享语义）
  再往后：RNN → LSTM → Transformer → LLM，本质上都在做同一件事：
    更准确地建模 P(下一个 token | 前面所有 token)
""")
