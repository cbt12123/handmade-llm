"""
08 · 真实生态：HuggingFace Transformers 上的真实模型
======================================================
第 4 章前面 7 节全部手写；这一节换成真实库，并做一件最有价值的事：
**把手写的注意力与 HF 模型内部的注意力做数值对照**，验证你写的是对的。

内容：
  1. 加载真实的 GPT-2 结构（tiny 版，几 MB），打印 config 与模块树
  2. 用模型真实的 c_attn / c_proj 权重跑一遍手写注意力，与 HF 的 attentions 对比
  3. 真实生成（greedy / top-p / 温度）+ KV Cache 的真实形状变化
  4. 只读 config（不下载权重）看真实大模型：Qwen2.5 / Mistral / Llama 的 GQA 配置

需要联网下载模型；离线时 [SKIP]。
"""
import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import check_close, human_num, require_module, section, set_seed, subsection, timer

set_seed(0)
tf = require_module("transformers", "第 4 章前 7 节是纯手写实现", pip_name="transformers")
torch = require_module("torch", "需要 torch", pip_name="torch")
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

MODEL = "hf-internal-testing/tiny-random-gpt2"

# ----------------------------------------------------------------------------------
section("1) 加载真实模型")
try:
    # 必须用与模型配套的那个 tokenizer（tiny 模型词表只有 1000，真实 gpt2 词表是 50257）
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # attn_implementation="eager" 才能拿到注意力权重（sdpa/flash 后端不返回）
    model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager")
    model.eval()
except Exception as e:
    print(f"[SKIP] 无法加载模型（{type(e).__name__}）：离线环境可跳过本节")
    raise SystemExit(0)

cfg = model.config
print(f"  模型: {MODEL}")
print(f"  n_layer={cfg.n_layer}  n_head={cfg.n_head}  n_embd={cfg.n_embd}  vocab={cfg.vocab_size}")
print(f"  参数量 = {human_num(float(sum(p.numel() for p in model.parameters())))}")
print("\n  模块树（第 0 层）：")
for name, _ in list(model.transformer.h[0].named_children()):
    print(f"    transformer.h.0.{name}")

# ----------------------------------------------------------------------------------
section("2) 手写注意力 vs HF 内部注意力（用真实权重）")


def manual_gpt2_attention(block, hidden, n_head):
    """用 GPT-2 第 0 层的真实权重，手算一遍注意力，返回 (out, attn_weights)。
    GPT-2 用 Conv1D：weight 形状是 (in, out)，所以前向是 x @ W + b（不是 W @ x）。"""
    d = hidden.shape[-1]
    dh = d // n_head
    attn = block.attn
    qkv = hidden @ attn.c_attn.weight + attn.c_attn.bias        # (T, 3d)
    q, k, v = qkv.chunk(3, dim=-1)
    T = hidden.shape[0]

    def split_heads(x):
        return x.view(T, n_head, dh).transpose(0, 1)             # (h, T, dh)

    q, k, v = split_heads(q), split_heads(k), split_heads(v)
    scores = (q @ k.transpose(-2, -1)) / (dh ** 0.5)             # ← 缩放点积
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    w = torch.softmax(scores, dim=-1)                            # ← 这就是注意力权重
    ctx = (w @ v).transpose(0, 1).reshape(T, d)
    out = ctx @ attn.c_proj.weight + attn.c_proj.bias
    return out, w


ids = torch.tensor([[5, 137, 921, 402, 88]])
block0 = model.transformer.h[0]
with torch.no_grad():
    out_hf = model(ids, output_attentions=True, output_hidden_states=True)
    emb = out_hf.hidden_states[0][0]                    # (T, d)：embedding 之后
    h_in = block0.ln_1(emb)                             # ← 注意力的输入要先过 ln_1（Pre-LN）
    attn_hf = out_hf.attentions[0][0]                   # (h, T, T)

out_manual, attn_manual = manual_gpt2_attention(block0, h_in, cfg.n_head)
print(f"  HF 注意力权重形状    : {tuple(attn_hf.shape)}")
print(f"  手写注意力权重形状   : {tuple(attn_manual.shape)}")
check_close(attn_manual, attn_hf, tol=1e-5, name="手写注意力 vs HF 内部注意力")
print("""
  注意力权重一致 = 算法一致。完整的 GPT-2 block 是
      emb + Attention(LN(emb)) + FFN(LN(emb + Attention))
  把残差与 FFN 补上就是第 4 章第 4 节的 TransformerBlock —— 区别只在于工程实现。
  （注意 GPT-2 用 Conv1D 而不是 nn.Linear：权重形状是 (in, out)，前向写成 x @ W + b）
""")
print("""
  这一步的意义：第 4 章手写的所有公式，在真实模型里就是这么算的。
  以后看到任何新架构，都可以用同样的方法"拆一层出来对照"。
""")

# ----------------------------------------------------------------------------------
section("3) 真实生成 + KV Cache 的真实形状")
prompt = "The capital of France is"
enc = tok(prompt, return_tensors="pt")
print(f"  prompt: {prompt!r} → {enc['input_ids'].shape[1]} 个 token")

with torch.no_grad():
    greedy = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
    print(f"  greedy  : {tok.decode(greedy[0][enc['input_ids'].shape[1]:])!r}")

    torch.manual_seed(0)
    sampled = model.generate(**enc, max_new_tokens=8, do_sample=True, top_p=0.9,
                             temperature=0.8, pad_token_id=tok.eos_token_id)
    print(f"  top-p   : {tok.decode(sampled[0][enc['input_ids'].shape[1]:])!r}")
print("  （tiny 随机权重模型没有语义，这里演示的是解码流程本身）")

print("\n  KV Cache 随生成步增长的实测：")
with torch.no_grad():
    cache = None
    ids_cur = enc["input_ids"]
    for step in range(4):
        o = model(ids_cur, use_cache=True, past_key_values=cache)
        cache = o.past_key_values
        k0 = cache.layers[0].keys if hasattr(cache.layers[0], "keys") else None
        shape = tuple(k0.shape) if k0 is not None else "(见下)"
        print(f"    step {step}: 输入 {ids_cur.shape[1]} token → 缓存 K 形状 {shape}")
        ids_cur = o.logits[:, -1:].argmax(-1, keepdim=True)
print("""
  第 1 维就是序列长度：每生成一步 +1，而每一步只需要喂 1 个 token
  → 这就是第 5 章第 3 节 KV Cache 的真实形态。
  在现代 transformers 里它是 DynamicCache（v4.36+），不再是可下标的 tuple。
""")

# ----------------------------------------------------------------------------------
section("4) 只读 config，看真实大模型的架构参数")
print(f"  {'模型':<28}{'层':>4}{'Q头':>5}{'KV头':>6}{'hidden':>8}{'词表':>10}{'上下文':>9}")
for name in ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-7B", "mistralai/Mistral-7B-v0.1",
             "t5-small", "google-bert/bert-base-uncased"]:
    try:
        c = AutoConfig.from_pretrained(name)
        print(f"  {name:<28}{getattr(c,'num_hidden_layers','-')!s:>4}"
              f"{getattr(c,'num_attention_heads','-')!s:>5}"
              f"{getattr(c,'num_key_value_heads','-')!s:>6}"
              f"{getattr(c,'hidden_size', getattr(c,'n_embd','-'))!s:>8}"
              f"{getattr(c,'vocab_size','-')!s:>10}"
              f"{str(getattr(c,'max_position_embeddings','-')):>9}")
    except Exception as e:
        print(f"  {name:<28} 读取失败：{type(e).__name__}")
print("""
  KV 头数 < Q 头数 = GQA（第 6 章第 4/7 节）：Qwen2.5-7B 是 28 Q 头 / 4 KV 头。
  注意：这里只读配置（几 KB），不下载几十 GB 的权重 —— 想了解架构时非常省事。
""")

section("5) 真实项目里怎么用")
print("""
  from transformers import AutoModelForCausalLM, AutoTokenizer
  model = AutoModelForCausalLM.from_pretrained(
      "Qwen/Qwen2.5-7B-Instruct",
      torch_dtype=torch.bfloat16,        # 省一半显存；A100/H100 建议 bf16
      device_map="auto",                 # accelerate 自动分配到多卡
      attn_implementation="flash_attention_2",   # 需要 GPU + 安装 flash-attn
  )
  生成：
  out = model.generate(**tok(messages, return_tensors="pt").to("cuda"),
                       max_new_tokens=512, temperature=0.7, top_p=0.9,
                       do_sample=True, use_cache=True)   # ← use_cache 默认就是 True
  国内网络：export HF_ENDPOINT=https://hf-mirror.com 或用 modelscope 下载。
""")
