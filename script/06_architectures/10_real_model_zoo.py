"""
10 · 真实生态：在 HuggingFace 上"拆"真实大模型（只读配置，不下载权重）
============================================================================
第 6 章前面 9 节逐个实现了各种架构；这一节去看**真实发布出来的模型**到底长什么样。

技巧：`AutoConfig.from_pretrained()` 只下载几 KB 的 config.json，
不用下载几十 GB 的权重 —— 想了解架构时非常省事。

本节会看到：GQA 的真实配置、MoE 的专家数、DeepSeek 的 MLA、Mamba 的 SSM 参数……
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common_utils import require_module, section, set_seed, subsection

set_seed(0)
tf = require_module("transformers", "第 6 章前 9 节是手写实现", pip_name="transformers")
from transformers import AutoConfig

# ----------------------------------------------------------------------------------
section("1) 主流开源模型的真实配置")
MODELS = [
    ("Qwen/Qwen2.5-7B", "Decoder-only（中文强，GQA 4 KV 头）"),
    ("Qwen/Qwen2.5-0.5B", "Decoder-only（小模型，GQA 2 KV 头）"),
    ("mistralai/Mistral-7B-v0.1", "Decoder-only（滑动窗口注意力）"),
    ("mistralai/Mixtral-8x7B-v0.1", "MoE（8 专家选 2）"),
    ("deepseek-ai/DeepSeek-V3", "MoE + MLA（256 专家，低秩 KV）"),
    ("state-spaces/mamba-2.8b-hf", "SSM（Mamba，非注意力）"),
    ("google-bert/bert-base-uncased", "Encoder-only（BERT）"),
    ("t5-small", "Encoder-Decoder（T5）"),
    ("openai/clip-vit-base-patch32", "多模态（CLIP 双编码器）"),
    ("google/vit-base-patch16-224", "视觉（ViT）"),
]

FIELDS = [
    ("层数", "num_hidden_layers", "n_layer"),
    ("Q 头", "num_attention_heads", "n_head"),
    ("KV 头", "num_key_value_heads", None),
    ("hidden", "hidden_size", "n_embd"),
    ("词表", "vocab_size", None),
    ("上下文", "max_position_embeddings", None),
]

print(f"  {'模型':<34}{'层':>4}{'Q头':>5}{'KV头':>6}{'hidden':>8}{'词表':>9}{'上下文':>9}")
print("  " + "-" * 74)
ok_models = []
for name, desc in MODELS:
    try:
        c = AutoConfig.from_pretrained(name)
    except Exception as e:
        print(f"  {name:<34} 读取失败（{type(e).__name__}）：可能需要登录或已被移除")
        continue
    ok_models.append((name, desc, c))
    vals = []
    for _label, f1, f2 in FIELDS:
        v = getattr(c, f1, None)
        if v is None and f2:
            v = getattr(c, f2, None)
        vals.append(str(v) if v is not None else "-")
    print(f"  {name:<34}" + "".join(f"{v:>{w}}" for v, w in
                                    zip(vals, [4, 5, 6, 8, 9, 9])))
print("""
  读表要点：
    · KV 头 < Q 头 就是 GQA（Qwen2.5-7B: 28 Q / 4 KV；Mistral-7B: 32 Q / 8 KV）
    · Meta 的 Llama 系列需要接受协议 + HF_TOKEN 才能访问 → 见第 5 节
""")

# ----------------------------------------------------------------------------------
section("2) 特殊架构的真实字段")
EXTRA = {
    "Mixtral-8x7B-v0.1": (["num_local_experts", "num_experts_per_tok"], "MoE：专家总数 / 每个 token 选几个"),
    "DeepSeek-V3": (["n_routed_experts", "num_experts_per_tok", "kv_lora_rank", "q_lora_rank"],
                    "MoE + MLA：kv_lora_rank 就是 MLA 压缩后的潜向量维度"),
    "Mistral-7B-v0.1": (["sliding_window"], "滑动窗口大小（第 6 章第 7 节）"),
    "mamba-2.8b-hf": (["state_size", "conv_kernel", "expand"], "SSM：状态维度 / 卷积核 / 扩展倍数"),
    "clip-vit-base-patch32": (["projection_dim", "text_config", "vision_config"], "CLIP：双编码器 + 投影维度"),
}
for name, (fields, note) in EXTRA.items():
    match = [m for m in ok_models if name.split("-")[0].lower() in m[0].lower()]
    if not match:
        continue
    c = match[0][2]
    print(f"\n  {match[0][0]}   ← {note}")
    for f in fields:
        v = getattr(c, f, None)
        if isinstance(v, object) and hasattr(v, "to_dict") and not isinstance(v, (int, float, str)):
            v = f"<{type(v).__name__} hidden={getattr(v,'hidden_size','?')}>"
        print(f"    {f:<22}{str(v)[:60]}")

# ----------------------------------------------------------------------------------
section("3) 真实模型的模块名（以 Qwen2.5-0.5B 为例）")
try:
    from transformers import AutoModelForCausalLM
    import torch

    m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.float32)
    layer0 = m.model.layers[0]
    print("  第 0 层的子模块：")
    for n, _ in layer0.named_children():
        print(f"    {n}")
    print(f"\n  注意力内部：{list(dict(layer0.self_attn.named_children()).keys())}")
    print(f"  MLP 内部  ：{list(dict(layer0.mlp.named_children()).keys())}")
    print("""
  对照第 6 章第 4 节：
    input_layernorm / post_attention_layernorm → RMSNorm（不是 LayerNorm！）
    self_attn.{q_proj,k_proj,v_proj}          → k_proj/v_proj 的输出维度 = KV 头数 × head_dim
    mlp.{gate_proj,up_proj,down_proj}         → 三个矩阵 = SwiGLU
  想确认是不是 RMSNorm：type(m.model.layers[0].input_layernorm).__name__
""")
    print(f"  归一化类型 = {type(layer0.input_layernorm).__name__}")
    print(f"  q_proj: {tuple(layer0.self_attn.q_proj.weight.shape)}   "
          f"k_proj: {tuple(layer0.self_attn.k_proj.weight.shape)}   ← 后者更小 = GQA")
except Exception as e:
    print(f"  [SKIP] 无法加载权重（{type(e).__name__}）：离线时只读 config 也可以")

# ----------------------------------------------------------------------------------
section("4) 怎么『看』一个不认识的新架构")
print("""
  ① AutoConfig 先读配置：层数/头数/dim/特殊字段（几十 KB，秒开）
  ② 看 config.json 里的 `architectures` 字段 → 去 transformers 源码搜同名类
  ③ 加载 tiny 版本（hf-internal-testing/tiny-random-xxx）看模块树
  ④ 读论文时按第 6 章的"五件套"对照：
     归一化 / 位置编码 / FFN 激活 / 注意力变体 / 是否 MoE
  ⑤ 跑一个最小的前向，打印每一步的形状（最快的理解方式）
""")

section("5) 访问受限模型（Llama / Gemma 等）")
print("""
  这些仓库需要先在 HuggingFace 网页上同意协议，然后带上 token：
      huggingface-cli login           # 或在代码里 token="hf_xxx"
      export HF_TOKEN=hf_xxx
  国内网络加速：
      export HF_ENDPOINT=https://hf-mirror.com
      # 或用 modelscope：pip install modelscope && modelscope download --model xxx
  离线部署：
      model.save_pretrained("./local_dir")  → 之后 from_pretrained("./local_dir")
""")
