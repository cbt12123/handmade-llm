"""
04 · HMM 词性标注 + Viterbi 解码：Transformer 之前，NLP 是概率图的天下
======================================================================
HMM 的两个假设：
    P(tag 序列) = Π P(tag_i | tag_{i-1})          一阶马尔可夫
    P(word | tag 序列) = Π P(word_i | tag_i)       观测独立

解码：在所有可能的 tag 序列中找概率最大的那条 —— 用动态规划（Viterbi）在 O(T·N²) 内完成，
而不是暴力搜索 N^T 条路径。

这个"用 DP 求最优路径"的思想，今天的 LLM 里依然在：CTC、CRF 解码层、beam search 都是亲戚。
"""
import sys
from pathlib import Path
from collections import defaultdict
import math

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common_utils import section, set_seed, subsection

set_seed(9)

# ----------------------------------------------------------------------------------
# 玩具标注语料（词/词性）
# ----------------------------------------------------------------------------------
TRAIN = [
    "the/DET dog/NOUN barks/VERB ./.",
    "a/DET cat/NOUN sleeps/VERB ./.",
    "the/DET dog/NOUN runs/VERB ./.",
    "my/DET friend/NOUN reads/VERB books/NOUN ./.",
    "she/PRON reads/VERB books/NOUN ./.",
    "he/PRON likes/VERB dogs/NOUN ./.",
    "the/DET cat/NOUN eats/VERB fish/NOUN ./.",
    "a/DET bird/NOUN sings/VERB ./.",
    "the/DET bird/NOUN flies/VERB ./.",
    "i/PRON like/VERB cats/NOUN ./.",
    "the/DET teacher/NOUN explains/VERB rules/NOUN ./.",
    "we/PRON explain/VERB rules/NOUN ./.",
]
TEST = [
    "the/DET cat/NOUN sleeps/VERB ./.",
    "he/PRON reads/VERB books/NOUN ./.",
    "a/DET teacher/NOUN explains/VERB rules/NOUN ./.",
]


def parse(sents):
    out = []
    for s in sents:
        pairs = [tuple(p.rsplit("/", 1)) for p in s.split()]
        out.append(pairs)
    return out


train = parse(TRAIN)
test = parse(TEST)
TAGS = sorted({t for s in train for _, t in s})
WORDS = sorted({w for s in train for w, _ in s})
N, Vw = len(TAGS), len(WORDS)
print(f"词性集合 ({N}): {TAGS}")
print(f"训练词表大小 = {Vw}")

# ----------------------------------------------------------------------------------
# 统计：转移计数 trans[t_{i-1}][t_i]，发射计数 emit[t][w]，初始计数 start[t]
# ----------------------------------------------------------------------------------
trans = defaultdict(lambda: defaultdict(int))
emit = defaultdict(lambda: defaultdict(int))
start = defaultdict(int)
for sent in train:
    start[sent[0][1]] += 1
    for i, (w, t) in enumerate(sent):
        emit[t][w] += 1
        if i > 0:
            trans[sent[i - 1][1]][t] += 1
tag_count = defaultdict(int)
for sent in train:
    for _, t in sent:
        tag_count[t] += 1

section("1) 极大似然估计 + add-1 平滑（全部转成对数，避免下溢）")
def log_p_trans(t_prev, t):
    return math.log((trans[t_prev][t] + 1) / (tag_count[t_prev] + N))


def log_p_emit(t, w):
    # 未登录词：分子为 1（平滑项），等价于给个很小的均匀概率
    return math.log((emit[t][w] + 1) / (tag_count[t] + Vw + 1))


def log_p_start(t):
    return math.log((start[t] + 1) / (len(train) + N))


print(f"  log P(NOUN|DET) = {log_p_trans('DET', 'NOUN'):.3f}   (P={math.exp(log_p_trans('DET', 'NOUN')):.3f})")
print(f"  log P(dog|NOUN) = {log_p_emit('NOUN', 'dog'):.3f}")
print(f"  log P(dog|VERB) = {log_p_emit('VERB', 'dog'):.3f}   ← VERB 后接 dog 的概率低得多")

section("2) Viterbi 解码（动态规划）")
def viterbi(words):
    T = len(words)
    dp = [{} for _ in range(T)]        # dp[t][tag] = 到 t 时刻、以 tag 结尾的最优路径得分
    bp = [{} for _ in range(T)]        # bp[t][tag] = 该路径上 t−1 时刻的 tag
    for t, tag in enumerate(TAGS):     # 初始化
        dp[0][tag] = log_p_start(tag) + log_p_emit(tag, words[0])
        bp[0][tag] = None
    for i in range(1, T):              # 递推
        for cur in TAGS:
            best_score, best_prev = -1e18, None
            for prev in TAGS:
                s = dp[i - 1][prev] + log_p_trans(prev, cur) + log_p_emit(cur, words[i])
                if s > best_score:
                    best_score, best_prev = s, prev
            dp[i][cur] = best_score
            bp[i][cur] = best_prev
    last = max(TAGS, key=lambda tg: dp[T - 1][tg])
    path = [last]
    for i in range(T - 1, 0, -1):      # 回溯
        path.append(bp[i][path[-1]])
    return list(reversed(path)), dp[T - 1][last]


print("  状态数 N=4、句长 T=5 时：暴力搜索 4⁵=1024 条路径，Viterbi 只需 5×4×4=80 次比较。")

section("3) 在测试句上评估")
correct = total = 0
for sent in test:
    words = [w for w, _ in sent]
    gold = [t for _, t in sent]
    pred, score = viterbi(words)
    hit = sum(int(p == g) for p, g in zip(pred, gold))
    correct += hit
    total += len(gold)
    print(f"  句子: {' '.join(words)}")
    print(f"    正确: {gold}")
    print(f"    预测: {pred}   (log 概率={score:.2f})")
print(f"\n  词性标注准确率 = {correct}/{total} = {correct / total:.4f}")

section("4) 换个角度：Viterbi 就是『带权 DAG 上的最短路』")
print("""
  时刻 t 的每个词性是一个节点，节点间边权 = log P(转移) + log P(发射)
  Viterbi = 逐层求"到每个节点的最大累计得分"
  这套 DP 后来演化成：
    · CRF 的 forward-backward（BiLSTM-CRF 曾是 NER 的 SOTA）
    · CTC 的前向算法（语音识别）
    · LLM 解码阶段的 beam search（第 5 章）—— 思路同构，只是"状态"换成了 token 序列
""")

section("5) 统计模型 → 神经网络的分水岭")
print("""
  HMM 的优点：可解释、小数据可用、解码有全局最优保证
  HMM 的缺点：
    ① 每个词只有"词形"这一个特征（无法利用前缀/后缀/上下文词）
    ② 一阶马尔可夫假设太弱（"not good" 的否定无法建模）
    ③ 发射概率无法处理未登录词的语义
  神经网络（BiLSTM / BERT / LLM）用"上下文相关的向量"彻底解决了 ①③，
  代价是失去了可解释性和概率保证 —— 这个 trade-off 贯穿整个 NLP 发展史。
""")
