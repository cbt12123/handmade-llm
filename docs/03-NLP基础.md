# 第 3 章 · NLP 基础：从分词到 Attention

> 配套代码：`script/03_nlp/`
> 主线：**怎么把"符号"（文字）变成"可微分的向量"，并用它建模 P(下一个 token | 前文)**。

---

## 3.1 分词：LLM 的隐形天花板

脚本：`script/03_nlp/01_tokenization_bpe.py`

三种粒度：

| 粒度 | 例子 | 问题 |
|---|---|---|
| char | `"hello" → h e l l o` | 词表极小但序列极长 |
| word | `"I love dogs" → I love dogs` | **OOV**（未登录词） |
| **subword (BPE)** | `"unhappiness" → un happi ness` | LLM 通用方案 |

### BPE 训练算法（GPT-2 / Llama 用的就是它）

```python
from collections import defaultdict, Counter

def get_vocab(corpus):
    """初始词表：每个单词拆成字符 + 结尾标记 </w>"""
    vocab = Counter()
    for line in corpus:
        for word in line.split():
            vocab[" ".join(list(word)) + " </w>"] += 1
    return vocab

def get_pair_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        syms = word.split()
        for i in range(len(syms) - 1):
            pairs[(syms[i], syms[i+1])] += freq
    return pairs

def merge_pair(pair, vocab_in):
    return {w.replace(" ".join(pair), "".join(pair)): f for w, f in vocab_in.items()}

vocab = get_vocab(corpus)
merges = []                                          # ← 有序的合并规则，这就是 BPE 的"模型"
for step in range(num_merges):
    pairs = get_pair_stats(vocab)
    best = max(pairs, key=pairs.get)                 # 每次合并出现最多的那一对
    vocab = merge_pair(best, vocab)
    merges.append(best)
```

推理时按学习顺序依次应用规则：

```python
def bpe_encode(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:                              # 顺序不能乱！
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i+1] == b:
                symbols[i:i+2] = [a + b]
            else:
                i += 1
    return symbols
```

脚本的真实输出：

```
merge  0: s + </w>  -> s</w>      (出现 9 次)
merge  1: t + h      -> th         (出现 6 次)
merge  3: th + e</w> -> the</w>    (出现 5 次)   ← 第 3 步就把 "the" 合成了一个 token
merge  9: do + g      -> dog        (出现 4 次)

dog          -> ['dog</w>']
dogs         -> ['dog', 's</w>']
dogcatcher   -> ['dog', 'c', 'a', 't', 'c', 'h', 'er</w>']   ← 没见过也能拆
zebra        -> ['z', 'e', 'b', 'r', 'a', '</w>']            ← 全是单字符，没有 OOV
```

```text
while 还没到目标词表大小:
    统计所有相邻符号对的出现次数
    把出现次数最多的那一对合并成一个新符号，记录这条规则   ← 合并顺序就是 BPE 的"模型参数"
```

推理时**按学习顺序依次应用**这些规则：

```
"dogcatcher" -> ['dog', 'catch', 'er']        # 训练时没见过这个词，但片段都认识
```

### 为什么 LLM 数不清 "strawberry 里有几个 r"

因为它**看到的是 token 序列，不是字母序列**。数字母、反转单词、首字母缩写、
按字母排序这类任务都踩在同一个坑里。

### 三种主流分词

| 方案 | 合并准则 | 代表 |
|---|---|---|
| **BPE** | 合并出现频次最高的相邻对 | GPT / Llama / Qwen |
| WordPiece | 合并"使语言模型似然增益最大"的对 | BERT |
| Unigram | 从大词表开始删掉"损失最小"的 token | SentencePiece / T5 |

> 现代实现跑在 **UTF-8 字节**上（byte-level BPE），所以任何语言/emoji 都不会 OOV。
>
> 词表大小是三角权衡：词表大 → 序列短（省算力）→ 但 embedding 与输出层参数暴增。
> 经验：让"平均每个 token ≈ 3~4 个字符"。GPT-2 50k、Llama-3 128k、Qwen 150k+。

---

## 3.2 n-gram 语言模型

马尔可夫假设：`P(wᵢ | w₁..wᵢ₋₁) ≈ P(wᵢ | wᵢ₋ₙ₊₁..wᵢ₋₁)`

三个必须解决的问题：

1. **概率为 0**：语料里没出现的组合 → 整句概率变 0
2. **平滑**：add-k、线性插值（用低阶统计兜底）、Kneser-Ney
3. **评价**：`perplexity = exp(−(1/N)Σ log P(wᵢ|context))` —— 这个指标一直用到今天的 LLM

平滑三件套（全部可手算）：

```python
import math
V = len(vocab)

def p_mle(w2, w1):                                   # 无平滑：没见过就是 0，整句概率变 0
    return bigram[(w1, w2)] / unigram[w1] if unigram[w1] else 0.0

def p_addk(w2, w1, k=0.5):                           # add-k：给每个计数加 k
    return (bigram[(w1, w2)] + k) / (unigram[w1] + k * V)

def p_interp(w3, w2, w1, l1=0.2, l2=0.3, l3=0.5):    # 线性插值：用低阶统计兜底
    p3 = trigram[(w1, w2, w3)] / bigram[(w1, w2)] if bigram[(w1, w2)] else 0.0
    p2 = bigram[(w2, w3)] / unigram[w2] if unigram[w2] else 0.0
    p1 = unigram[w3] / total_tokens
    return l3 * p3 + l2 * p2 + l1 * p1
```

困惑度（LLM 时代仍在用的指标）：

```python
def perplexity(sent, smooth_fn):
    sent = ["<s>"] + list(sent) + ["</s>"]
    logp, cnt = 0.0, 0
    for i in range(1, len(sent)):
        p = max(smooth_fn(sent[i], sent[i-1]), 1e-10)
        logp += math.log(p)
        cnt += 1
    return math.exp(-logp / cnt)                     # = exp(平均交叉熵)
```

脚本输出：

```
bigram + add-k(0.5) 的 perplexity = 4.86
bigram 无平滑       的 perplexity = 1.80
```

> ⚠️ 先别急着下结论：这里无平滑"更好"是因为测试句 `i like deep learning`
> 本身就出现在训练语料里（背答案）。只要换一句含未见组合的句子，
> 无平滑的 perplexity 会**直接变成 ∞**（乘进一个 0 概率项），平滑才是有意义的比较。

生成（就按概率采样下一个词）：

```python
def generate(start="<s>", max_len=10):
    out, cur = [], start
    for _ in range(max_len):
        cands = [(w2, c) for (w1, w2), c in bigram.items() if w1 == cur]
        if not cands: break
        words, weights = zip(*cands)
        nxt = rng.choices(words, weights=weights)[0]
        if nxt == "</s>": break
        out.append(nxt); cur = nxt
    return " ".join(out)
```

```
-> books
-> i like machine learning are fun
-> deep learning needs data and compute      ← 词汇连贯、语法破碎，n-gram 的典型特征
```

脚本：`script/03_nlp/02_ngram_language_model.py`

**为什么被神经网络取代**：窗口只有 n−1 个词，参数随 n 指数爆炸（Vⁿ），
且无法泛化到没见过的组合。神经网络把词映射成稠密向量后，参数与词表**线性**相关。

---

## 3.3 Word2Vec：分布式语义假设

负采样的更新（纯 numpy，每步只有几行）：

```python
LR, NEG = 0.1, 8
probs = counts ** 0.75                               # 负采样分布：词频的 3/4 次方
probs /= probs.sum()

for center, context in pairs:
    vc = W_in[center]                                # 中心词向量
    # ---- 正样本 ----
    uo = W_out[context]
    g_pos = sigmoid(float(vc @ uo)) - 1.0            # dL/d(v·u)
    grad_vc = g_pos * uo
    W_out[context] -= LR * g_pos * vc
    # ---- 负样本（向量化，别写 Python 循环）----
    negs = rng.choice(V, size=NEG, p=probs)
    Un = W_out[negs]                                 # (K, D)
    g_neg = sigmoid(Un @ vc)
    grad_vc += g_neg @ Un
    np.add.at(W_out, negs, -LR * g_neg[:, None] * vc)
    W_in[center] -= LR * grad_vc
```

为什么 `np.add.at` 而不是 `W_out[negs] -= ...`？
因为一个 batch 里可能抽到**重复的负样本**，普通索引赋值只会保留最后一次，`add.at` 才真正累加。

训练后的效果（脚本真实输出）：

```
cat    -> bird(1.00), dog(0.99), mouse(0.98), fish(0.97), bed(0.66)
king   -> man(0.99), woman(0.98), queen(0.98), crown(0.63), hat(0.63)
eats   -> feeds(0.70), chases(0.63), sleeps(0.47), runs(0.44), rules(0.42)

king − man + woman -> [('queen', 0.952), ('crown', 0.67), ('hat', 0.67)]
```

类比的实现只有三行：

```python
def analogy(a, b, c, topn=3):
    q = W_in[w2i[b]] - W_in[w2i[a]] + W_in[w2i[c]]   # a:b = c:?  →  v(?) ≈ v(b) − v(a) + v(c)
    sims = [(cosine(q, W_in[i]), i2w[i]) for i in range(V) if i not in (w2i[a], w2i[b], w2i[c])]
    return sorted(sims, reverse=True)[:topn]
```

> 注：负采样的 loss **不保证单调下降**（随机近似 + 向量范数增长）。
> 评估词向量要看最近邻与类比，不要盯着 loss。

脚本：`script/03_nlp/03_word2vec_numpy.py`（纯 numpy 实现 Skip-gram + 负采样）

> "You shall know a word by the company it keeps."

```
目标：最大化 P(context | center)
负采样近似：loss = −log σ(v_c·u_o) − Σ_{k=1..K} log σ(−v_c·u_{n_k})
```

**为什么需要负采样**：原始 softmax 分母要对全部 V 个词求和 → 每个样本 O(V)；
负采样把它变成 K+1 个二分类 → 每个样本 O(K)，K 通常 5~20。
负采样分布取词频的 **3/4 次方**（抑制高频词）。

> 向量的加减真的对应语义维度 —— "性别轴"和"王权轴"被学成了两个近似正交的方向。

**局限 → 下一代的动机**：`bank`（银行/河岸）只有一个向量，无法消歧。
解决方案是**上下文相关表示**（ELMo → BERT → LLM），见第 6 章第 1 节。

---

## 3.4 HMM 词性标注 + Viterbi

参数估计（add-1 平滑，全部转对数防止下溢）：

```python
import math

def log_p_trans(t_prev, t):  return math.log((trans[t_prev][t] + 1) / (tag_count[t_prev] + N))
def log_p_emit(t, w):        return math.log((emit[t][w]     + 1) / (tag_count[t]      + Vw + 1))
def log_p_start(t):          return math.log((start[t]       + 1) / (len(train)        + N))

print(f"log P(NOUN|DET) = {log_p_trans('DET','NOUN'):.3f}")
print(f"log P(dog|NOUN) = {log_p_emit('NOUN','dog'):.3f}")     # 高
print(f"log P(dog|VERB) = {log_p_emit('VERB','dog'):.3f}")     # 低 → 这就是判别依据
```

Viterbi 解码（动态规划，注意 `bp` 记录路径）：

```python
def viterbi(words):
    T = len(words)
    dp = [{} for _ in range(T)]          # dp[t][tag] = 到 t 时刻、以 tag 结尾的最优得分
    bp = [{} for _ in range(T)]
    for tag in TAGS:
        dp[0][tag] = log_p_start(tag) + log_p_emit(tag, words[0])
    for i in range(1, T):
        for cur in TAGS:
            best_score, best_prev = -1e18, None
            for prev in TAGS:
                s = dp[i-1][prev] + log_p_trans(prev, cur) + log_p_emit(cur, words[i])
                if s > best_score:
                    best_score, best_prev = s, prev
            dp[i][cur], bp[i][cur] = best_score, best_prev
    last = max(TAGS, key=lambda t: dp[T-1][t])
    path = [last]
    for i in range(T-1, 0, -1):          # 回溯
        path.append(bp[i][path[-1]])
    return list(reversed(path))
```

脚本输出：

```
句子: the cat sleeps .
  正确: ['DET', 'NOUN', 'VERB', '.']
  预测: ['DET', 'NOUN', 'VERB', '.']   (log 概率=-11.61)
词性标注准确率 = 13/13 = 1.0000
```

脚本：`script/03_nlp/04_hmm_viterbi_pos.py`

两个假设：

```
P(tag 序列)   = Π P(tag_i | tag_{i-1})      一阶马尔可夫
P(word | tag) = Π P(word_i | tag_i)         观测独立
```

Viterbi = **带权 DAG 上的最短路**（动态规划）：

```
dp[t][tag] = max_{prev} ( dp[t-1][prev] + log P(tag|prev) + log P(word_t|tag) )
bp[t][tag] = argmax 的那个 prev
最后从 dp[T-1] 的最大值回溯出整条路径
```

复杂度从暴力的 `N^T` 降到 `O(T·N²)`。脚本在玩具语料上达到较高准确率（可运行查看）。

**这套 DP 思想后来演化成了**：CRF 的 forward-backward（BiLSTM-CRF 曾是 NER SOTA）、
CTC（语音识别）、以及 **beam search**（第 5 章第 2 节，思路同构，状态换成了 token 序列）。

### 统计模型 vs 神经网络的分水岭

| HMM 优点 | HMM 缺点 |
|---|---|
| 可解释、小数据可用、解码有全局最优保证 | 只用"词形"一个特征（无法用前后缀/上下文） |
| | 一阶马尔可夫太弱（"not good" 的否定无法建模） |
| | 发射概率无法处理未登录词的语义 |

神经网络解决了后两点，代价是失去可解释性与概率保证 —— 这个 trade-off 贯穿整个 NLP 史。

---

## 3.5 RNN / LSTM 与梯度消失

把梯度真的回传一遍（不是比喻，是数值）：

```python
# ---- RNN：h_t = tanh(W_xh·x_t + W_hh·h_{t-1} + b) ----
g = 2.0 * hs[-1].copy()                              # 取 L = ‖h_T‖²，则 ∂L/∂h_T = 2h_T
for t in range(T-2, -1, -1):
    # ∂L/∂h_t = W_hhᵀ · diag(tanh'(a_{t+1})) · ∂L/∂h_{t+1}    ← 每步都要乘一次 W_hh
    g = W_hh.T @ ((1 - np.tanh(a_s[t+1])**2) * g)
    grads.append(np.linalg.norm(g))

# ---- LSTM 的细胞状态：c_t = f_t⊙c_{t-1} + i_t⊙g_t ----
g_c = 2.0 * cs[-1].copy()
for t in range(T-2, -1, -1):
    g_c = fs[t+1] * g_c                              # ← 只乘遗忘门（≈1），没有矩阵连乘
    grads.append(np.linalg.norm(g_c))
```

脚本的真实输出（序列长 30）：

```
模型      t=0 处梯度 / t=T−1 处梯度
RNN                      8.109e-08     ← 30 步之前的信息几乎完全丢失
LSTM                     1.591e-04     ← 高出 3 个数量级（c 路径只乘遗忘门 f≈1）
```

LSTM 单步展开（前向，背下来）：

```python
gates = W_x @ x_t + W_h @ h + b
i = sigmoid(gates[:DH])            # 输入门
f = sigmoid(gates[DH:2*DH])        # 遗忘门（初始化偏置设为 1：先别忘）
g = np.tanh(gates[2*DH:3*DH])      # 候选
o = sigmoid(gates[3*DH:])          # 输出门
c = f * c + i * g                  # ← 关键：加法更新，不是反复乘矩阵
h = o * np.tanh(c)
```

脚本：`script/03_nlp/05_rnn_lstm_numpy.py`（**数值实验，不是比喻**）

```
RNN:   h_t = tanh(W_xh·x_t + W_hh·h_{t-1} + b)
       ∂L/∂h_t = W_hhᵀ · diag(tanh'(a_{t+1})) · ∂L/∂h_{t+1}      ← 每步都乘一次 W_hh

LSTM:  c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t          ← 加法！
       ∂L/∂c_t = f_{t+1} ⊙ ∂L/∂c_{t+1}          ← 只乘遗忘门（≈1）
```

实测（序列长 30）：

| 模型 | `‖∂L/∂h_0‖ / ‖∂L/∂h_T‖` |
|---|---|
| RNN | ~1e-8（指数衰减） |
| LSTM（c 路径） | ~1e-1（几乎不衰减） |

**但 LSTM 并没有赢到最后**：它仍然要**串行**跑 T 步，无法并行。
这个缺陷直接催生了 Transformer —— 把"沿时间递归"换成"一次矩阵乘法"。

---

## 3.6 Seq2Seq + Attention：最后一级台阶

三种打分函数（第三种就是 Transformer 用的那个）：

```python
import numpy as np

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def additive_scores(dec_h, enc_hs):        # Bahdanau：先线性变换，过 tanh，再点乘 v
    return np.tanh(dec_h @ W1.T + enc_hs @ W2.T) @ v

def dot_scores(dec_h, enc_hs):             # Luong：直接点积，无参数
    return enc_hs @ dec_h

def scaled_dot_scores(dec_h, enc_hs):      # Transformer：点积 / √d
    return enc_hs @ dec_h / np.sqrt(D)
```

一次解码步（就这三行）：

```python
def decode_step(dec_h, enc_hs, score_fn=scaled_dot_scores):
    a = softmax(score_fn(dec_h, enc_hs))       # ① 分配注意力
    context = a @ enc_hs                       # ② 按权重把 value 加权平均
    return a, context
```

脚本的真实输出（`Q·K/√d` 版本）：

```
加性 additive      α = [0.390 0.234 0.185 0.191]   最关注: i
点积 dot           α = [0.    0.    1.    0.   ]   最关注: machine
缩放点积 scaled    α = [0.010 0.010 0.956 0.023]   最关注: machine

生成 我    ← 注意力 [0.875 0.080 0.021 0.024]
生成 爱    ← 注意力 [0.194 0.677 0.026 0.103]
生成 机器  ← 注意力 [0.010 0.010 0.956 0.023]      ← 对齐正确
生成 学习  ← 注意力 [0.010 0.019 0.018 0.954]
```

> 点积版本直接给出近乎 one-hot 的权重（softmax 饱和），
> 缩放版本保留了"软"分配 —— 这正是第 4 章除以 √d 的动机。

脚本：`script/03_nlp/06_seq2seq_attention.py`

原始 Seq2Seq 把整句压成一个向量（瓶颈问题）；
Bahdanau Attention 让解码每一步都"回头看"源句：

```
score(h_dec, h_enc)        加性：vᵀ·tanh(W₁·h_dec + W₂·h_enc)
α      = softmax(score)
context= Σ αᵢ · h_encᵢ
```

三种打分函数：

| 类型 | 公式 | 用在哪 |
|---|---|---|
| 加性 (Bahdanau) | `vᵀ tanh(W₁q + W₂k)` | 原始 Seq2Seq |
| 点积 (Luong) | `q·k` | 简单，无参数 |
| **缩放点积** | `q·k / √d` | **Transformer 用的就是这个** |

**为什么要除以 √d**（第 4 章最关键的一句话）：
点积的方差随 d 线性增长 → softmax 输入量级过大 → 梯度饱和。除以 √d 把方差拉回 1。

### 本节 → 下一章的映射表

| 本节概念 | Transformer 里的对应物 |
|---|---|
| `h_enc_i`（编码器状态） | **Key / Value** |
| `h_dec`（解码器状态） | **Query** |
| `score(q, k)` | `QKᵀ / √d` |
| `Σαᵢ·h_i` | `softmax(QKᵀ/√d)·V` |
| 单个注意力头 | 多头注意力（并行 h 次再拼接） |

> **Attention is All You Need = 把本节这套机制从"每秒一步"变成"一次矩阵乘法"。**

---

## 3.7 速记卡

```
· 分词决定 LLM 的"视野"：BPE 是现代标准，中文 1 token ≈ 1~1.5 汉字
· 语言模型的唯一目标：P(下一个 token | 前文)，从 n-gram 到 LLM 都没变
· perplexity = exp(交叉熵)，沿用至今
· Word2Vec 用负采样把 V 分类变成 K+1 个二分类；静态向量无法消歧
· Viterbi = 带权 DAG 最短路；beam search 是它的"token 序列版"
· RNN 梯度指数衰减（每步乘 W_hh）；LSTM 用加法路径（只乘 f_t）缓解
· 但 LSTM 串行 → 被 Transformer 取代
· attention = 可微分的字典查询：Q 查询、K 索引、V 内容、softmax 是软 argmax
```

---

## 3.8 真实生态：HuggingFace tokenizers + datasets

```bash
python3 script/03_nlp/07_real_tokenizers.py    # pip install transformers
python3 script/03_nlp/08_real_datasets.py      # pip install datasets transformers
```

### 真实 tokenizer

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")       # BPE，词表 50257
tok.pad_token = tok.eos_token                     # GPT-2 原生没有 pad token！
ids  = tok("hello world").input_ids               # [31373, 995]
toks = tok.tokenize("strawberry")                 # ['st', 'raw', 'berry']

qwen = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")   # 中文友好，词表 151936
```

真实输出（脚本 `07_real_tokenizers.py`）：

```
strawberry     → ['st', 'raw', 'berry']   (3 个 token)
strawberries   → ['st', 'raw', 'berries'] (3 个 token)
raspberry      → ['r', 'aspberry']        (2 个 token)

文本              字符数   GPT-2 tokens   Qwen tokens
英文                 60            10            10
中文                 19            43            11      ← 老词表贵 4 倍！
代码                 73            32            20

批量编码 input_ids 形状: (2, 4)     attention_mask: [[1,1,0,0], [1,1,1,1]]
```

> 中文 19 个字，GPT-2 要 43 个 token，Qwen 只要 11 个 —— 做中文 RAG 时，
> 选 tokenizer 直接决定你的 embedding 与生成成本。

### 真实数据集 + DataLoader

```python
from datasets import load_dataset            # 新版要用完整的 "组织/数据集" 形式
ds = load_dataset("google-research-datasets/poem_sentiment")
ds_tok = ds.map(lambda b: tok(b["verse_text"], truncation=True, max_length=64), batched=True)

def collate(features):                        # 动态 padding：每个 batch 只 pad 到本批最长
    max_len = max(len(f["input_ids"]) for f in features)
    ...
loader = DataLoader(ds_tok["train"], batch_size=8, shuffle=True, collate_fn=collate)
```

真实输出（脚本 `08_real_datasets.py`）：

```
数据集: google-research-datasets/poem_sentiment
splits: ['train', 'validation', 'test']     train 样本数 = 892
第 0 条样本: verse_text='with pale blue berries. in these peaceful shades--'  label=1
[tokenize 全量] 556.10 ms
batch: input_ids (8, 9)   attention_mask (8, 9)   labels (8,)
本批最长序列 = 9（固定 padding 会一律 pad 到 64 → 浪费算力）
```

真实项目里 90% 的训练用 `Trainer`（脚本里给了完整配置：bf16、梯度累积、warmup、
余弦衰减、gradient_checkpointing、按 epoch 评估与存盘）。

---

## 3.9 代码索引

| 脚本 | 内容 |
|---|---|
| `01_tokenization_bpe.py` | 手写 BPE 训练 + 编码新词 + 三种分词对比 |
| `02_ngram_language_model.py` | bigram/trigram + add-k/插值平滑 + perplexity + 生成 |
| `03_word2vec_numpy.py` | 纯 numpy 的 Skip-gram + 负采样 + 类比实验 |
| `04_hmm_viterbi_pos.py` | HMM 参数估计 + Viterbi 解码 + 准确率评估 |
| `05_rnn_lstm_numpy.py` | RNN/LSTM 前向 + 梯度范数衰减的数值实验 |
| `06_seq2seq_attention.py` | 三种打分函数 + 对齐矩阵可视化 + 到 Transformer 的映射 |

```bash
python3 script/03_nlp/06_seq2seq_attention.py   # 推荐先跑这个（能看到对齐矩阵）
bash script/run_all.sh 03                        # 或跑完本章全部 6 个脚本
```
