"""
01 · 分词（Tokenization）：NLP 的第一道门，也是 LLM 的"隐形天花板"
====================================================================
三种粒度：
  char   : "hello" → ["h","e","l","l","o"] 词表极小但序列极长
  word   : "I love dogs" → ["I","love","dogs"] 直观但有 OOV（未登录词）
  subword: "unhappiness" → ["un","happi","ness"]  ← LLM 通用方案（BPE / WordPiece / Unigram）

本脚本从零实现 BPE（Byte Pair Encoding）—— GPT-2 / Llama 用的就是它。
理解 BPE，你就能理解为什么 LLM 算不对 "strawberry 里有几个 r"。
"""
import sys
from pathlib import Path
from collections import Counter, defaultdict

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common_utils import section, set_seed, subsection

set_seed(0)

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the dog barks and the fox runs",
    "a quick dog jumps higher than a lazy fox",
    "dogs and foxes are animals",
    "the quickest runner jumps over hurdles",
]


# ----------------------------------------------------------------------------------
section("1) 三种粒度的直观对比")
sample = "I love dogs and dogcatchers"
print(f"  原文: {sample}")
print(f"  char  : {list(sample.replace(' ', '_'))[:20]} ...  词表约 100，序列长 {len(sample)}")
print(f"  word  : {sample.split()}   问题：'dogcatchers' 几乎不会在训练语料里 → OOV")
print("  subword: 把罕见词拆成常见片段 → 既能覆盖新词，又不至于序列太长")

# ----------------------------------------------------------------------------------
section("2) 手写 BPE 训练")
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
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_pair(pair, vocab_in):
    """把所有出现该相邻对的地方合并成新符号。"""
    vocab_out = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for word, freq in vocab_in.items():
        vocab_out[word.replace(bigram, replacement)] = freq
    return vocab_out


def train_bpe(corpus, num_merges=20, verbose=True):
    vocab = get_vocab(corpus)
    merges = []                                        # 有序的合并规则，就是 BPE 的"模型"
    for step in range(num_merges):
        pairs = get_pair_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_pair(best, vocab)
        merges.append(best)
        if verbose and step < 12:
            print(f"  merge {step:>2}: {best[0]} + {best[1]} -> {''.join(best)}   (出现 {pairs[best]} 次)")
    return vocab, merges


vocab, merges = train_bpe(CORPUS, num_merges=30)
print(f"\n  学习到 {len(merges)} 条合并规则，最终 token 片段数 = "
      f"{len({s for w in vocab for s in w.split()})}")

subsection("3) 用学到的规则编码新词（推理时分词的真实流程）")
def bpe_encode(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:                                # 按学习顺序依次应用规则
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols[i:i + 2] = [a + b]
            else:
                i += 1
    return symbols


for w in ["dog", "dogs", "dogcatcher", "quickest", "zebra"]:
    print(f"  {w:<12} -> {bpe_encode(w, merges)}")
print("  注意 'dogcatcher' 这种训练时没见过的词，被拆成了已知片段：没有 OOV。")

section("4) 词表大小 / 序列长度 / 压缩率 的三角关系")
print("""
  · 词表越大 → 序列越短（省算力、上下文能装更多）→ 但 embedding + 输出层参数暴增
  · LLM 典型词表：GPT-2 50k、Llama-3 128k、Qwen 150k+
  · 经验公式：大模型词表 ≈ 使"平均每个 token ≈ 3~4 个字符"的那个点
  · 中文的坑：早期模型按字切（1 字 1 token），现代模型（Qwen/Llama3）把常见词合成一个 token
""")

subsection("5) 为什么 LLM 数不清 'strawberry' 里有几个 r")
tok = bpe_encode("strawberry", merges)
print(f"  BPE 切分: {tok}")
print("  模型『看到』的是 token 序列而不是字母序列，数字母任务需要它把 token 再拆开 —— 它做不到。")
print("  同类问题：反转单词、首字母缩写、按字母排序，都是 tokenization 带来的『盲区』。")

section("6) 现代 LLM 的三种主流分词")
print("""
  BPE      : 从字符开始反复合并最高频的相邻对（GPT / Llama / Qwen）
  WordPiece: 合并"使语言模型似然增益最大"的相邻对（BERT）
  Unigram  : 从大词表开始删掉"删了损失最小"的 token（SentencePiece / Llama 早期 / T5）
  补充：现代实现跑在 UTF-8 字节上（byte-level BPE），所以任何语言/emoji 都不会 OOV。
""")
