"""
02 · 解码策略：temperature / top-k / top-p / beam / 重复惩罚
============================================================
模型输出的是 logits → softmax → 下一个 token 的概率分布。
"怎么从这个分布里选一个 token" 就是解码策略，它直接决定生成文本的质量与多样性。

本脚本全部手写：
  ① greedy（贪心）        每次选概率最大的 → 确定但容易重复、套话
  ② temperature          把 logits 除以 T：T<1 更尖锐、T>1 更平滑
  ③ top-k                只在概率最高的 k 个里采样
  ④ top-p (nucleus)      在累积概率达到 p 的最小集合里采样（动态 k）
  ⑤ 重复惩罚             对已出现过的 token 降权（frequency / presence penalty）
  ⑥ beam search          保留 m 条候选路径（翻译/摘要等"求最优"任务更合适）
"""
import sys
from pathlib import Path
from collections import defaultdict
import math

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import section, set_seed, subsection

set_seed(11)

# ----------------------------------------------------------------------------------
# 用一个玩具 bigram 语言模型当作"被采样的 LLM"
# ----------------------------------------------------------------------------------
CORPUS = [
    "the model learns patterns from data",
    "the model generates text token by token",
    "data drives the model behaviour",
    "tokens are the basic units of language models",
    "language models learn from large data",
]
bigram = defaultdict(lambda: defaultdict(float))
for line in CORPUS:
    ws = ["<s>"] + line.split() + ["</s>"]
    for a, b in zip(ws[:-1], ws[1:]):
        bigram[a][b] += 1
vocab = sorted({w for d in bigram.values() for w in d} | {"<s>"})
V = len(vocab)
w2i = {w: i for i, w in enumerate(vocab)}
i2w = {i: w for w, i in w2i.items()}


def logits_of(word):
    """玩具 LM：给定上一个词，返回下一个词的 logits"""
    counts = bigram.get(word, {})
    z = np.zeros(V)
    for w, c in counts.items():
        z[w2i[w]] = math.log(c + 1)
    z += np.random.default_rng(abs(hash(word)) % (2**32)).normal(0, 0.35, V)   # 加点随机性
    return z


def probs(logits, temperature=1.0):
    z = logits / max(temperature, 1e-6)
    e = np.exp(z - z.max())
    return e / e.sum()


# ----------------------------------------------------------------------------------
section("1) 温度对分布的塑形（同一个上下文 'the'）")
z = logits_of("the")
for T in [0.3, 0.7, 1.0, 1.5, 2.0]:
    p = probs(z, T)
    top = np.argsort(-p)[:5]
    ent = -np.sum(p * np.log(p + 1e-12))
    print(f"  T={T:<4} 熵={ent:.2f}  top5: " +
          ", ".join(f"{i2w[i]}({p[i]:.3f})" for i in top))
print("""
  温度的本质：给 logits 做缩放。T→0 变成贪心，T→∞ 变成均匀分布。
  代码/ChatGPT 场景常用 0.2~0.8；创意写作用 0.8~1.2。
""")

# ----------------------------------------------------------------------------------
section("2) 五种采样策略的实现")
rng = np.random.default_rng(0)


def sample_greedy(z):
    return int(np.argmax(z))


def sample_temperature(z, temperature=1.0):
    return int(rng.choice(V, p=probs(z, temperature)))


def sample_top_k(z, k=5, temperature=1.0):
    p = probs(z, temperature)
    idx = np.argsort(-p)[:k]
    p_k = p[idx] / p[idx].sum()
    return int(rng.choice(idx, p=p_k))


def sample_top_p(z, p_thresh=0.9, temperature=1.0):
    """nucleus sampling：按概率从高到低累加，直到达到 p"""
    p = probs(z, temperature)
    order = np.argsort(-p)
    cumsum = np.cumsum(p[order])
    cutoff = int(np.searchsorted(cumsum, p_thresh) + 1)
    idx = order[:cutoff]
    p_nuc = p[idx] / p[idx].sum()
    return int(rng.choice(idx, p=p_nuc))


def sample_with_penalty(z, history, presence=0.5, frequency=0.5, temperature=1.0):
    """重复惩罚：对已出现的 token 从 logits 里直接减掉惩罚项（ OpenAI API 的做法）"""
    z2 = z.copy()
    for w, cnt in history.items():
        z2[w2i[w]] -= presence + frequency * cnt
    return int(rng.choice(V, p=probs(z2, temperature)))


def generate(strategy, length=8, seed=0):
    global rng
    rng = np.random.default_rng(seed)
    cur = "<s>"
    out, history = [], defaultdict(int)
    for _ in range(length):
        z = logits_of(cur)
        if strategy == "greedy":
            nxt = sample_greedy(z)
        elif strategy == "temperature":
            nxt = sample_temperature(z, 1.2)
        elif strategy == "top_k":
            nxt = sample_top_k(z, k=3, temperature=1.0)
        elif strategy == "top_p":
            nxt = sample_top_p(z, p_thresh=0.85, temperature=1.0)
        else:
            nxt = sample_with_penalty(z, history, 0.6, 0.3, 0.9)
        if nxt == w2i["</s>"]:
            break
        word = i2w[nxt]
        out.append(word)
        history[word] += 1
        cur = word
    return " ".join(out)


print("  贪心      :", generate("greedy", seed=1))
print("  T=1.2     :", generate("temperature", seed=1))
print("  top-k=3   :", generate("top_k", seed=1))
print("  top-p=0.85:", generate("top_p", seed=1))
print("  带重复惩罚 :", generate("penalty", seed=1))

section("3) top-p 与 top-k 的区别（为什么 top-p 更常用）")
z = logits_of("model")
p = probs(z, 1.0)
order = np.argsort(-p)
print("  排序后的概率:", np.round(p[order][:8], 4))
print(f"  top-k=5  固定保留前 5 个，累积概率 = {p[order[:5]].sum():.3f}")
for thr in [0.5, 0.8, 0.95]:
    cum = np.cumsum(p[order])
    k = int(np.searchsorted(cum, thr) + 1)
    print(f"  top-p={thr}  动态保留前 {k} 个，累积概率 = {p[order[:k]].sum():.3f}")
print("""
  当分布很尖锐时（模型很确定），top-p 只取 1~2 个候选；分布平坦时自动扩大候选集。
  这是 top-p 比 top-k 更"自适应"的原因 —— 也是它成为默认选项的原因。
""")

section("4) Beam Search：什么时候该用它")
def beam_search(length=6, beam_width=3):
    beams = [([], 0.0, "<s>")]                     # (已生成词, log 概率, 上一个词)
    for _ in range(length):
        cands = []
        for words, score, cur in beams:
            p = probs(logits_of(cur), 1.0)
            for i in np.argsort(-p)[:6]:
                if i2w[i] == "</s>":
                    continue
                cands.append((words + [i2w[i]], score + math.log(p[i] + 1e-12), i2w[i]))
        beams = sorted(cands, key=lambda x: -x[1])[:beam_width]
    return beams


print(f"  beam width=3 的 top-3 结果：")
for words, score, _ in beam_search():
    print(f"    logP={score:>8.3f}  {' '.join(words)}")
print("""
  特点：beam search 求的是"整句联合概率最大"，比贪心更全局，但代价是
    · 生成缺乏多样性（多条 beam 常常只差一两个词）
    · 容易产出"安全但无聊"的句子（开放式生成里反而更差）
  适用：翻译、摘要、结构化抽取这类有明确参考答案的任务。
""")

section("5) 生产环境常用参数组合")
print("""
  通用对话      : temperature=0.7, top_p=0.9, repetition_penalty≈1.05
  代码/数学      : temperature=0.2, top_p=0.95（要稳定、要可复现）
  创意写作      : temperature=1.0~1.2, top_p=0.95, presence_penalty=0.3~0.6
  JSON/函数调用  : temperature=0 + grammar-constrained decoding（约束解码，只允许合法 token）
  注意：temperature=0 与 greedy 在浮点层面不完全等价（仍受并行 kernel 的归约顺序影响），
        真要严格复现需要固定 seed、batch、kernel 版本。
""")
