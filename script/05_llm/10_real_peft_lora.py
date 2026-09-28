"""
10 · 真实生态：用 peft 做真实 LoRA 微调
==========================================
第 5 章第 7 节手写了 LoRA；真实项目用 `peft` 库（+ `transformers` Trainer 或自己的循环）。

本节完整走一遍真实流程：
  加载真实模型 → 配置 LoRA → 看可训练参数占比 → 训几步 → 合并/保存/加载 adapter

需要联网下载模型（Qwen2.5-0.5B，约 1GB）；离线时 [SKIP]。
"""
import sys
from pathlib import Path
import tempfile

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import human_num, require_module, section, set_seed, subsection, timer

set_seed(0)
peft = require_module("peft", "第 5 章第 7 节是手写实现", pip_name="peft")
tf = require_module("transformers", "需要 transformers", pip_name="transformers")
torch = require_module("torch", "需要 torch", pip_name="torch")
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel

MODEL = "Qwen/Qwen2.5-0.5B"
try:
    with timer(f"加载 {MODEL}"):
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()
except Exception as e:
    print(f"[SKIP] 无法加载模型（{type(e).__name__}）：本节需要联网下载 {MODEL}")
    raise SystemExit(0)

if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# ----------------------------------------------------------------------------------
section("1) 冻结全部权重，插入 LoRA")
base_params = sum(p.numel() for p in model.parameters())
print(f"  基座模型: {MODEL}   参数量 = {human_num(float(base_params))}")

config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,                                     # 低秩维度
    lora_alpha=16,                           # scaling = alpha / r = 2
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],     # 原论文的选择；实践中常把所有线性层都加上
    bias="none",
)
model_lora = get_peft_model(model, config)
model_lora.print_trainable_parameters()
trainable = sum(p.numel() for p in model_lora.parameters() if p.requires_grad)
print(f"  占比 = {trainable / base_params:.4%}   （rank 越大越多，见第 5 章第 7 节的表）")

section("2) target_modules 的选择对参数量的影响")
for mods in [["q_proj"], ["q_proj", "v_proj"], ["q_proj", "k_proj", "v_proj", "o_proj"],
             ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]]:
    cfg2 = LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16, target_modules=mods)
    m2 = get_peft_model(AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32), cfg2)
    n2 = sum(p.numel() for p in m2.parameters() if p.requires_grad)
    print(f"  {str(mods)[:52]:<54}{n2:>12,} 参数  ({n2 / base_params:.3%})")
    del m2
print("""
  · 只加 q/v ：最省，够用（原论文配置）
  · 加全部注意力矩阵：常用折中
  · 再加 FFN 三个矩阵：效果最好，参数约翻倍（仍然不到 1%）
""")

# ----------------------------------------------------------------------------------
section("3) 训几步看看（真实的前向+反向）")
texts = [
    "Machine learning models learn patterns from data.",
    "LoRA makes fine-tuning large models much cheaper.",
    "The key idea is that updates are low rank.",
    "Only a small number of parameters are trainable.",
]
enc = tok(texts, return_tensors="pt", padding=True, truncation=True)
input_ids, attn = enc["input_ids"], enc["attention_mask"]
labels = input_ids.clone()
labels[attn == 0] = -100                     # ← padding 不参与 loss

opt = torch.optim.AdamW([p for p in model_lora.parameters() if p.requires_grad], lr=1e-4)
model_lora.train()
for step in range(6):
    out = model_lora(input_ids=input_ids, attention_mask=attn, labels=labels)
    opt.zero_grad()
    out.loss.backward()
    opt.step()
    print(f"  step {step}: loss = {out.loss.item():.4f}")
print("""
  真实微调的要点：
    · lr 比全量微调高一个量级（1e-4 ~ 2e-4）
    · labels 里把 padding 设成 -100（与第 4 章第 5 节的做法一致）
    · 通常还要加 warmup + 余弦衰减 + 梯度裁剪
""")

# ----------------------------------------------------------------------------------
section("4) 保存的 adapter 有多小")
with tempfile.TemporaryDirectory() as tmp:
    model_lora.save_pretrained(tmp)
    size = sum(f.stat().st_size for f in Path(tmp).rglob("*") if f.is_file())
    print(f"  adapter 文件大小 = {size / 1024:.0f} KB")
    print(f"  基座模型大小     ≈ {base_params * 4 / 1024**2:.0f} MB (fp32)")
    print(f"  → 一个基座 + 一堆 adapter = 多任务共享，切换成本几乎为零")

    # 重新加载：基座 + adapter
    base2 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    loaded = PeftModel.from_pretrained(base2, tmp)
    print(f"  重新加载后可训练参数 = "
          f"{sum(p.numel() for p in loaded.parameters() if p.requires_grad):,}")
print("""
  这就是 LoRA 在工程中最大的价值：
    · 存储：4bit 基座 + 几十 MB 的 adapter（而不是每个任务一份完整模型）
    · 切换：同一个基座上秒级加载不同 adapter
    · 合并：merge_and_unload() 后部署时零额外延迟
""")

section("5) merge 与 unload（部署用）")
merged = model_lora.merge_and_unload()          # 把 LoRA 权重并回基座
print(f"  合并后模型类型: {type(merged).__name__}")
print(f"  参数量 = {human_num(float(sum(p.numel() for p in merged.parameters())))} "
      f"（与基座一致，没有额外模块）")
with torch.no_grad():
    a = merged(input_ids=input_ids, attention_mask=attn).logits
    b = model_lora(input_ids=input_ids, attention_mask=attn).logits
print(f"  合并前后 logits 最大差异 = {(a - b).abs().max().item():.2e}")
print("  差异为 0 → 部署时直接合并，推理速度与基座完全相同。")

section("6) 完整微调脚本骨架（真实项目照抄）")
print("""
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", dtype=torch.bfloat16, device_map="auto")
model.config.use_cache = False                  # 训练时必须关掉 KV Cache！

peft_config = LoraConfig(
    task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# trainable params: 40,370,176 || all params: 7,656,293,888 || trainable%: 0.5272

args = TrainingArguments(
    output_dir="out", per_device_train_batch_size=4, gradient_accumulation_steps=8,
    num_train_epochs=1, learning_rate=1e-4, warmup_ratio=0.05,
    lr_scheduler_type="cosine", logging_steps=10,
    bf16=True, gradient_checkpointing=True, save_strategy="steps", save_steps=200,
)
trainer = Trainer(model=model, args=args, train_dataset=tokenized_ds, data_collator=collator)
trainer.train()
model.save_pretrained("out/adapter")            # 只存 adapter
```

  ⚠️ 常见坑：
    ① 训练时忘了 `model.config.use_cache=False` → 显存爆掉
    ② 用 gradient_checkpointing 时要 `model.enable_input_require_grads()`
    ③ 数据格式：指令微调要用 chat template（`tok.apply_chat_template`），别手拼
""")
