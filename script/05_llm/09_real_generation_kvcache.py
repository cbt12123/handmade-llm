"""
09 · 真实生态：transformers 的 generate() 与真实 KV Cache
============================================================
第 5 章第 2、3 节手写了采样与 KV Cache；真实库把它们封装进了 `model.generate()`。
本节做三件事：
  1. 用 generate() 的真实参数跑一遍（temperature / top_p / top_k / beam / 重复惩罚）
  2. 实测 use_cache=True/False 的加速比与显存（真实模型上的数字）
  3. 打印 KV Cache 的真实字节数，验证第 5 章第 1 节的估算公式

需要联网下载模型；离线时 [SKIP]。
"""
import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import bytes_str, require_module, section, set_seed, subsection, timer

set_seed(0)
tf = require_module("transformers", "第 5 章前 8 节是手写实现", pip_name="transformers")
torch = require_module("torch", "需要 torch", pip_name="torch")
from transformers import AutoTokenizer, AutoModelForCausalLM

# 真实小模型：Qwen2.5-0.5B（约 1GB）。若下载失败，回退到 tiny 随机模型
CANDIDATES = ["Qwen/Qwen2.5-0.5B", "hf-internal-testing/tiny-random-gpt2"]
model = tok = None
for name in CANDIDATES:
    try:
        with timer(f"加载 {name}"):
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
        model.eval()
        print(f"  使用模型: {name}")
        break
    except Exception as e:
        print(f"  加载 {name} 失败：{type(e).__name__}（尝试下一个）")
if model is None:
    print("[SKIP] 无法下载模型，本节需要联网。")
    raise SystemExit(0)

if tok.pad_token is None:
    tok.pad_token = tok.eos_token
cfg = model.config
print(f"  层数={cfg.num_hidden_layers if hasattr(cfg,'num_hidden_layers') else cfg.n_layer}  "
      f"隐藏维度={cfg.hidden_size if hasattr(cfg,'hidden_size') else cfg.n_embd}")

prompt = "The quick brown fox jumps over the lazy dog. The dog"
enc = tok(prompt, return_tensors="pt")
print(f"\n  prompt: {prompt!r}")
print(f"  token 数: {enc['input_ids'].shape[1]}")

# ----------------------------------------------------------------------------------
section("1) generate() 的真实参数")
def gen(tag, **kw):
    with torch.no_grad():
        torch.manual_seed(0)
        out = model.generate(**enc, max_new_tokens=24, pad_token_id=tok.pad_token_id,
                             use_cache=True, **kw)
    text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"  {tag:<26}→ {text[:70]!r}")
    return out


gen("greedy", do_sample=False)
gen("temperature=0.7", do_sample=True, temperature=0.7)
gen("temperature=1.5", do_sample=True, temperature=1.5)
gen("top_p=0.9", do_sample=True, temperature=0.8, top_p=0.9)
gen("top_k=10", do_sample=True, temperature=0.8, top_k=10)
gen("repetition_penalty=1.5", do_sample=False, repetition_penalty=1.5)
gen("beam=3", do_sample=False, num_beams=3, early_stopping=True)

print("""
  参数含义（第 5 章第 2 节的手写实现与此一一对应）：
    temperature        : logits / T，越小越确定
    top_p / top_k      : 截断低概率尾部（nucleus / 固定个数）
    repetition_penalty : 已出现过的 token 除以惩罚系数（>1 抑制复读）
    num_beams          : beam search，最大化整句概率（翻译/摘要更合适）
""")

# ----------------------------------------------------------------------------------
section("2) use_cache 的真实加速比")
def timed_generate(use_cache, n_new=40, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        torch.manual_seed(0)
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(**enc, max_new_tokens=n_new, do_sample=False,
                           use_cache=use_cache, pad_token_id=tok.pad_token_id)
        best = min(best, time.perf_counter() - t0)
    return best


t_cache = timed_generate(True)
t_nocache = timed_generate(False)
print(f"  use_cache=True : {t_cache * 1000:8.1f} ms （生成 40 个 token）")
print(f"  use_cache=False: {t_nocache * 1000:8.1f} ms")
print(f"  加速比         : {t_nocache / t_cache:.2f}×")
print("""
  注意：小模型 + CPU 上加速比有限（Python/调度开销占比大）；
  真实 7B 模型在 GPU 上通常是 5~20×，而且序列越长收益越大（O(T²) → O(T)）。
""")

# ----------------------------------------------------------------------------------
section("3) 真实 KV Cache 的字节数（验证第 5 章第 1 节的公式）")
with torch.no_grad():
    o = model(**enc, use_cache=True)
    cache = o.past_key_values
    k0 = cache.layers[0].keys
print(f"  第 0 层 K 形状: {tuple(k0.shape)}   dtype: {k0.dtype}")
n_layers = len(cache.layers)
n_kv_heads = k0.shape[1]
head_dim = k0.shape[-1]
seq = k0.shape[2]
dtype_bytes = k0.element_size()
total = sum(cache.layers[l].keys.numel() + cache.layers[l].values.numel()
            for l in range(n_layers)) * dtype_bytes
print(f"  层数={n_layers}  KV 头数={n_kv_heads}  head_dim={head_dim}  序列长度={seq}")
print(f"  实测 KV Cache 总大小 = {bytes_str(total)}")
formula = 2 * n_layers * seq * n_kv_heads * head_dim * dtype_bytes
print(f"  公式 2·L·T·kv_heads·d_head·bytes = {bytes_str(formula)}")
print(f"  每 token = {bytes_str(total // seq)}")
print("  公式与实测一致 → 第 5 章第 1 节的显存估算可以直接用。")

print("\n  不同上下文长度下的 KV Cache（本模型，fp32）：")
for s in [1024, 4096, 16384, 32768]:
    b = 2 * n_layers * s * n_kv_heads * head_dim * dtype_bytes
    print(f"    seq={s:>6}: {bytes_str(b)}")

section("4) token 预算与真实成本")
texts = {"英文 100 词": "Machine learning is a field of study in artificial intelligence "
                        "concerned with the development of statistical algorithms that can "
                        "learn from data and generalize to unseen data, and thus perform "
                        "tasks without explicit instructions. " * 2,
         "中文 100 字": "机器学习是人工智能的一个重要分支，它研究如何通过计算手段利用经验来"
                        "改善系统自身的性能。深度学习是机器学习的一个子集，它使用多层神经网络"
                        "来学习数据的表示。近年来，大语言模型在自然语言处理领域取得了巨大成功。" * 2}
print(f"  {'文本':<12}{'字符数':>8}{'token 数':>10}{'字符/token':>12}")
for name, s in texts.items():
    n = len(tok(s).input_ids)
    print(f"  {name:<12}{len(s):>8}{n:>10}{len(s) / n:>12.2f}")

section("5) 生产建议")
print("""
  ① 服务端别用 generate() 的默认配置：显式写 use_cache=True、do_sample、max_new_tokens
  ② 批量生成要 left padding（tokenizer.padding_side="left"），否则结果会错位
  ③ 长文本生成用 vLLM / TGI（连续批处理 + PagedAttention），吞吐差 5~10 倍
  ④ 真正上线前一定做「回归评测集」：50~200 条自有数据，量化/换模型都跑一遍
  ⑤ 记录 token 消耗：输入 token + 输出 token 都要算钱
""")
