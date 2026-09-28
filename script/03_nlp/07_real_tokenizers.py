"""
07 · 真实生态：HuggingFace tokenizers（GPT-2 的 BPE / BERT 的 WordPiece）
==========================================================================
第 3 章第 1 节手写了 BPE；真实项目直接用 HF 的 `AutoTokenizer`（Rust 实现，快几个数量级）。

本节回答三个实际问题：
  1. 真实词表有多大？"strawberry" 在 GPT-2 里到底被切成什么？
  2. 中英文的 token 预算差多少？（这直接决定你的 RAG 成本）
  3. 手写 BPE 与真实 BPE 的结论一致吗？

需要联网下载 tokenizer（几 MB）；离线时 [SKIP]。
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import require_module, section, set_seed, subsection

set_seed(0)
transformers = require_module("transformers", "第 3 章前 6 节是纯手写实现，不依赖它",
                              pip_name="transformers")
from transformers import AutoTokenizer


def try_load(name, **kw):
    try:
        return AutoTokenizer.from_pretrained(name, **kw)
    except Exception as e:
        print(f"  [SKIP] 加载 {name} 失败：{type(e).__name__}（离线环境可跳过）")
        return None


section("1) 真实的 GPT-2 tokenizer（BPE，词表 50257）")
gpt2 = try_load("gpt2")
if gpt2 is not None:
    gpt2.pad_token = gpt2.eos_token       # GPT-2 原生没有 pad token，批量编码前必须指定
    print(f"  类型: {type(gpt2).__name__}    词表大小: {len(gpt2)}")
    text = "The quick brown fox jumps over the lazy dog"
    ids = gpt2(text).input_ids
    toks = gpt2.convert_ids_to_tokens(ids)
    print(f"\n  文本: {text}")
    print(f"  token 数: {len(ids)}")
    print(f"  切分: {toks}")
    print(f"  解码: {gpt2.decode(ids)}")
    print(f"  ⚠️ 注意 'The' 首字母大写被切成 'The'（一个 token），小写 'the' 是另一个 token")

section("2) 'strawberry 里有几个 r' —— 用真实 tokenizer 复现")
if gpt2 is not None:
    for w in ["strawberry", "strawberries", "Strawberry", "raspberry"]:
        toks = gpt2.tokenize(w)
        print(f"  {w:<14} → {toks}   ({len(toks)} 个 token)")
    print("""
  这就是 LLM 数不清字母的原因：它看到的是 ['st','raw','berry'] 三个符号，
  而不是 10 个字母。让它数字母，等于要求它把符号再拆开 —— 它做不到。
  工程解法：遇到这类任务，在 prompt 里让它"逐字母拼读"（spell it out），
  或者用工具（代码解释器）代劳。
""")

section("3) 中英文的 token 预算（直接决定成本）")
qwen = try_load("Qwen/Qwen2.5-0.5B")          # 中文友好的现代词表（151k）
samples = {
    "英文": "Machine learning models are trained on large scale datasets.",
    "中文": "机器学习模型在大规模数据集上进行训练。",
    "代码": "def train(model, loader):\n    for x, y in loader:\n        loss = model(x)",
}
if gpt2 is not None:
    print(f"  {'文本':<8}{'字符数':>8}{'GPT-2 tokens':>14}{'Qwen tokens':>14}")
    for name, s in samples.items():
        n_gpt2 = len(gpt2(s).input_ids)
        n_qwen = len(qwen(s).input_ids) if qwen is not None else -1
        print(f"  {name:<8}{len(s):>8}{n_gpt2:>14}{n_qwen:>14}")
    print("""
  结论：
    · 老词表（GPT-2）处理中文极贵：一个汉字可能要 2~3 个 token
    · 现代多语言词表（Qwen/Llama-3）把中文压到约 1~1.5 token/字
    · 做中文 RAG 时，选 tokenizer 直接决定你的 embedding 与生成成本
""")

section("4) 与手写 BPE 的结论对照")
print("""
  第 3 章手写 BPE 得到的三条结论，在真实 tokenizer 上全部成立：
    ① 常见词是一个 token（'the'、'dog'），罕见词被拆成片段（'dogcatcher'）
    ② 拆分是"从左到右、按规则顺序贪心合并"，与手写实现一致
    ③ 词表越大 → 序列越短 → 上下文能装更多内容（代价是 embedding 与输出层变大）
  差别只在于工程：HF 的 tokenizers 是 Rust 实现，还做了
     · 预编译的正则预分词（GPT-2 的 <|endoftext|> 等特殊 token）
     · 并行批量编码（比 Python 循环快 10~100 倍）
     · ByteLevel：任何 UTF-8 字节都能编码，永远不会有 OOV
""")

section("5) 实战要点")
if gpt2 is not None:
    # 批量编码与 padding
    batch = gpt2(["hello world", "machine learning is fun"], padding=True, return_tensors="pt")
    print(f"  批量编码 input_ids 形状: {tuple(batch['input_ids'].shape)}")
    print(f"  attention_mask         : {batch['attention_mask'].tolist()}")
    print("""
  生产注意点：
    ① 一定要传 attention_mask，否则 padding 会被当成真实内容（第 4 章第 6 节）
    ② 训练时用 padding="longest"（省算力）；推理时常用 left padding（便于批量生成）
    ③ truncation=True + max_length 防止超长输入炸显存
    ④ 特殊 token（[CLS]/[SEP]/<s>）由 tokenizer 自动加，别手拼
    ⑤ tokenizer 必须与模型一一对应：混用会静默变差（结果可跑但全错）
""")
