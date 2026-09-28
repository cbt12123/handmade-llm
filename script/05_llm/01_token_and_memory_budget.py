"""
01 · LLM 的"算账"：token 预算、显存占用、训练成本估算
======================================================
做 LLM 工程的第一步永远是算账：
  · 一段文本有多少 token？上下文窗口够不够？
  · 推理要多少显存？（权重 + KV Cache + 激活）
  · 训练要多少显存？（权重 + 梯度 + 优化器状态 + 激活重计算）

本脚本把这些全部写成函数，你可以直接改参数估算自己的模型。
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common_utils import bytes_str, human_num, section, subsection

# ----------------------------------------------------------------------------------
section("1) Token 预算：上下文窗口到底能装多少")
print("""
经验系数（不同 tokenizer 差异很大，这里给常用近似值）：
  英文：1 token ≈ 4 个字符 ≈ 0.75 个单词
  中文：1 token ≈ 1 ~ 1.5 个汉字（现代词表友好的模型，如 Qwen）
  代码：1 token ≈ 3 ~ 4 个字符（缩进也算 token）
""")
CHARS_PER_TOKEN = {"en": 4.0, "zh": 1.2, "code": 3.5}
for name, ratio in CHARS_PER_TOKEN.items():
    for window in [4096, 8192, 32768, 128000]:
        n_chars = int(window * ratio)
        demo = "汉字" if name == "zh" else "x"
        print(f"  {name:<5} 窗口 {window:>7} token ≈ {n_chars:>8,} 个字符"
              f"  ≈ {n_chars // 500:>5} 段 500 字文本")
    print()

subsection("容易被忽略的事实：输出也算 token")
print("""
  · 上下文 = 输入 token + 已生成的输出 token（总长度受窗口限制）
  · 多轮对话会不断累积历史 → 8k 窗口可能 10 轮就满了 → 需要摘要/截断/滑动窗口
  · 系统提示词（system prompt）每轮都要重算 → 长 system prompt 的成本是"每轮"的
""")

# ----------------------------------------------------------------------------------
section("2) 推理显存 = 模型权重 + KV Cache + 少量激活")


def kv_cache_bytes(num_layers, num_kv_heads, head_dim, seq_len, batch=1, dtype_bytes=2):
    """KV Cache：每层每头每个位置存 K 和 V 两个向量。"""
    return num_layers * 2 * batch * seq_len * num_kv_heads * head_dim * dtype_bytes


def model_weight_bytes(num_params, dtype_bytes=2):
    return num_params * dtype_bytes


CONFIGS = {                       # (参数量, 层数, KV 头数, head_dim, d_model)
    "Llama-3-8B":   (8e9, 32, 8, 128, 4096),
    "Llama-3-70B":  (70e9, 80, 8, 128, 8192),
    "Qwen2-7B":     (7e9, 28, 4, 128, 3584),
    "Mixtral-8x7B": (46.7e9, 32, 8, 128, 4096),
}
print(f"{'模型':<14}{'权重(fp16)':>12}{'KV/token':>12}{'4k 上下文':>12}{'32k 上下文':>12}")
for name, (params, layers, kv_heads, hd, d) in CONFIGS.items():
    w = model_weight_bytes(params, 2)
    per_token = kv_cache_bytes(layers, kv_heads, hd, 1)
    print(f"  {name:<12}{bytes_str(w):>12}{bytes_str(per_token):>12}"
          f"{bytes_str(per_token * 4096):>12}{bytes_str(per_token * 32768):>12}")

print("""
  注意 KV 头数 ≠ query 头数：这是 GQA（分组查询注意力），第 6 章会讲。
  8B 模型、32k 上下文、batch=8 时，KV Cache 就能轻松超过权重本身 → 长上下文的主要瓶颈。
""")

# ----------------------------------------------------------------------------------
section("3) 训练显存：为什么 7B 模型要 8 张 80G 卡")


def training_bytes(num_params, dtype_bytes=2, optimizer="adamw", grad_accum_activation_gb=0.0):
    """混合精度训练的经典估算：
       权重 fp16 (2) + 梯度 fp16 (2) + fp32 主权重 (4) + Adam m (4) + Adam v (4) = 16 字节/参数"""
    bytes_per_param = 2 + 2 + 4 + 4 + 4          # 混合精度 + AdamW
    if optimizer == "sgd":
        bytes_per_param = 2 + 2 + 4              # 无动量、无 fp32 主副本
    elif optimizer == "8bit-adam":
        bytes_per_param = 2 + 2 + 4 + 1 + 1      # 8-bit 优化器状态
    return num_params * bytes_per_param + grad_accum_activation_gb * 1024**3


for p, name in [(7e9, "7B"), (13e9, "13B"), (70e9, "70B")]:
    adam = training_bytes(p)
    adam8 = training_bytes(p, optimizer="8bit-adam")
    print(f"  {name:<5} AdamW(fp16混精): {bytes_str(adam):>10}   8-bit 优化器: {bytes_str(adam8):>10}"
          f"   （还需额外算激活与激活重计算）")

print("""
  省显存三板斧：
    · ZeRO-1/2/3（把优化器状态 / 梯度 / 参数切分到多卡）
    · 梯度检查点（activation checkpointing，用 30% 额外算力换 5~10 倍激活显存）
    · 8-bit 优化器 / 低秩优化器（Adafactor、LOMO）
""")

# ----------------------------------------------------------------------------------
section("4) 训练算力估算（Chinchilla 风格）")


def training_flops(num_params, num_tokens):
    """经典近似：FLOPs ≈ 6 · N · D（前向 2N + 反向 4N）"""
    return 6 * num_params * num_tokens


print(f"{'模型':<8}{'训练 token':>14}{'FLOPs':>16}{'A100(312TFLOPs) 天':>20}")
for params, tokens, label in [
    (1e9, 20e9, "1B"), (7e9, 1e12, "7B"), (70e9, 2e12, "70B"), (70e9, 15e12, "70B(充分训练)"),
]:
    flops = training_flops(params, tokens)
    a100_flops = 312e12 * 0.4                     # 假设 40% 利用率
    days = flops / a100_flops / 86400
    print(f"  {label:<16}{human_num(tokens):>12}{human_num(flops):>16}{days:>18.1f}")

print("""
  经验法则（Chinchilla）：参数量 × 20 ≈ 最优训练 token 数
  现实中的 LLM 往往"超量训练"（Llama-3 8B 用了 15T token），因为推理成本比训练成本更值钱。
""")

section("5) 推理成本：prefill vs decode")
print("""
  两个阶段的计算特征完全不同：
    Prefill（处理输入）: 一次算完所有 token → 计算密集（compute-bound），可大 batch
    Decode （逐个生成）: 每次只算 1 个 token → 访存密集（memory-bound），权重反复读
  因此：
    · Prefill 看 FLOPs；Decode 看显存带宽 → 这就是为什么"量化"对 decode 加速特别明显
    · 提高吞吐的关键：连续批处理（continuous batching）+ KV Cache 复用（下一节）
""")
