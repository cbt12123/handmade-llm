"""
08 · 真实生态：datasets 库 + 真实数据集 + DataLoader
======================================================
前面几节用的都是合成语料；真实项目里数据是"脏"的，而且流程是：
    下载数据集 → 分词 → 动态 padding → DataLoader → 训练循环 / Trainer

本节用一个真实的小数据集（poem_sentiment：诗句情感分类）走完整条链路，
并给出 HF Trainer 的标准用法（真实项目里 90% 的训练代码长那样）。

需要联网下载数据集（约 1MB）；离线时 [SKIP]。
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import require_module, section, set_seed, subsection, timer

set_seed(0)
datasets = require_module("datasets", "第 3 章前 7 节不依赖它", pip_name="datasets")
torch = require_module("torch", "需要 torch", pip_name="torch")
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset

# ----------------------------------------------------------------------------------
section("1) 加载一个真实数据集")
# 注意：新版 datasets 要求用完整的 "组织/数据集" 形式
CANDIDATES = [("google-research-datasets/poem_sentiment", None),
              ("nyu-mll/glue", "sst2"),
              ("stanfordnlp/imdb", None)]
ds = None
used = None
for name, cfg in CANDIDATES:
    try:
        ds = load_dataset(name, cfg) if cfg else load_dataset(name)
        used = name
        break
    except Exception as e:
        print(f"  尝试 {name} 失败：{type(e).__name__}（换下一个）")
if ds is None:
    print("[SKIP] 无法下载数据集（离线环境）。本节其余部分需要联网，可跳过。")
    raise SystemExit(0)

print(f"  数据集: {used}")
print(f"  splits: {list(ds.keys())}")
print(f"  train 样本数 = {len(ds['train'])}")
print(f"  字段: {ds['train'].column_names}")
print(f"\n  第 0 条样本:")
for k, v in ds["train"][0].items():
    print(f"    {k}: {v}")

# ----------------------------------------------------------------------------------
section("2) 用真实 tokenizer 做 map（批量、并行）")
tf = require_module("transformers", "需要 transformers", pip_name="transformers")
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token


def tokenize_fn(batch):
    return tok(batch["verse_text"] if "verse_text" in batch else batch["text"],
               truncation=True, max_length=64)


text_col = "verse_text" if "verse_text" in ds["train"].column_names else (
    "sentence" if "sentence" in ds["train"].column_names else "text")
label_col = "label"
with timer("tokenize 全量（HF datasets 的 map 是缓存 + 并行的）"):
    ds_tok = ds.map(lambda b: tok(b[text_col], truncation=True, max_length=64), batched=True)
ds_tok = ds_tok.remove_columns([c for c in ds_tok["train"].column_names
                                if c not in ["input_ids", "attention_mask", label_col]])
ds_tok.set_format("torch")
print(f"  tokenize 后字段: {ds_tok['train'].column_names}")
print(f"  第 0 条的 input_ids 长度 = {len(ds_tok['train'][0]['input_ids'])}")

# ----------------------------------------------------------------------------------
section("3) 动态 padding 的 collate_fn（别再一次性 pad 到 max_length）")


def collate(features):
    max_len = max(len(f["input_ids"]) for f in features)       # ← 每个 batch 只 pad 到本批最长
    input_ids, masks, labels = [], [], []
    for f in features:
        ids = f["input_ids"]
        pad_n = max_len - len(ids)
        input_ids.append(F.pad(ids, (0, pad_n), value=tok.pad_token_id))
        masks.append(torch.cat([torch.ones(len(ids)), torch.zeros(pad_n)]))
        labels.append(f[label_col])
    return {"input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(masks),
            "labels": torch.tensor(labels)}


loader = DataLoader(ds_tok["train"], batch_size=8, shuffle=True, collate_fn=collate)
batch = next(iter(loader))
print(f"  batch: input_ids {tuple(batch['input_ids'].shape)}  "
      f"attention_mask {tuple(batch['attention_mask'].shape)}  labels {tuple(batch['labels'].shape)}")
print(f"  本批最长序列 = {batch['input_ids'].shape[1]}（固定 padding 会一律 pad 到 64 → 浪费算力）")

# ----------------------------------------------------------------------------------
section("4) 真实训练循环（用第 4 章手写的 Transformer 也完全一样）")
n_classes = len(set(ds_tok["train"][label_col]))
print(f"  类别数 = {n_classes}")


class TinyClassifier(nn.Module):
    """一个极简的文本分类器：embedding → 均值池化 → Linear"""
    def __init__(self, vocab=50257, dim=64, n_cls=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim, padding_idx=tok.pad_token_id)
        self.fc = nn.Linear(dim, n_cls)

    def forward(self, ids, mask):
        e = self.emb(ids)
        m = mask.unsqueeze(-1).float()
        pooled = (e * m).sum(1) / m.sum(1).clamp(min=1)      # ← 用 mask 做均值，别让 pad 参与
        return self.fc(pooled)


model = TinyClassifier(n_cls=n_classes)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for step, batch in enumerate(loader):
    logits = model(batch["input_ids"], batch["attention_mask"])
    loss = F.cross_entropy(logits, batch["labels"])
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 40 == 0:
        acc = (logits.argmax(1) == batch["labels"]).float().mean().item()
        print(f"  step {step:>3}  loss={loss.item():.4f}  batch acc={acc:.4f}")
    if step >= 120:
        break
print("\n  （这里只跑 120 步做演示；真实训练要跑多个 epoch + 验证集早停）")

# ----------------------------------------------------------------------------------
section("5) HF Trainer：真实项目里 90% 的训练代码长这样")
print("""
```python
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=n_classes)

args = TrainingArguments(
    output_dir="./out",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,        # 显存不够时的标准做法
    num_train_epochs=3,
    learning_rate=2e-5,                   # BERT 系微调的典型 lr
    warmup_ratio=0.1,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    fp16=True,                            # 或 bf16=True（A100/H100）
    gradient_checkpointing=True,          # 省激活显存
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",                     # 想看曲线就换成 "tensorboard"/"wandb"
)

trainer = Trainer(model=model, args=args,
                  train_dataset=ds_tok["train"], eval_dataset=ds_tok["test"],
                  data_collator=collate,
                  compute_metrics=lambda p: {"acc": (p.predictions.argmax(-1) == p.label_ids).mean()})
trainer.train()
```

Trainer 帮你做了：混合精度、梯度累积、调度器、检查点、日志、分布式（配合 accelerate）。
代价是"黑盒"—— 想改训练逻辑时，就用第 1 章那个手写循环 + accelerate 自己写。
""")

section("6) 数据集的工程经验")
print("""
  ① 先用 1% 数据跑通全流程（"`split='train[:1%]'`"），再全量跑
  ② map 有缓存：改了预处理函数要 `load_dataset(..., download_mode="force_redownload")` 或删缓存
  ③ 长尾样本（超长文本）要提前过滤，否则一个 batch 会被拉到 OOM
  ④ 类别不平衡：用 WeightedRandomSampler 或 class_weight
  ⑤ 数据版本管理：DVC / git-lfs / 直接固定 datasets 的 revision（保证可复现）
  ⑥ LLM 微调数据：质量 >> 数量，1k 条精心构造的数据常常赢过 100k 条爬来的数据
""")
