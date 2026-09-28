"""
00 · 大模型架构谱系：一图看清"谁是谁"
======================================
按【注意力可见性】划分，现代大模型只有三种基本范式：

  ┌──────────────┬────────────────────────┬───────────────────────────┐
  │ Encoder-only │ Encoder-Decoder        │ Decoder-only              │
  │ 双向可见      │ 编码器双向 + 解码器因果  │ 仅因果（只看左边）          │
  ├──────────────┼────────────────────────┼───────────────────────────┤
  │ BERT/RoBERTa │ T5 / BART / Transformer│ GPT / Llama / Qwen / Mistral│
  │ 适合理解任务  │ 适合翻译/摘要           │ 适合生成（LLM 主流）        │
  └──────────────┴────────────────────────┴───────────────────────────┘

再往后是"改良派"：
  · 稀疏化：MoE（Mixtral / DeepSeek / Switch）
  · 非注意力：SSM / Mamba / RWKV
  · 注意力变体：MQA / GQA / MLA / 滑动窗口 / 线性注意力

本脚本输出一张可对照的架构谱系表，并给出"选型建议"。
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common_utils import human_num, section, subsection

section("1) 三大范式对照")
ROWS = [
    ("范式", "Encoder-only", "Encoder-Decoder", "Decoder-only"),
    ("代表", "BERT / RoBERTa", "T5 / BART / 原版 Transformer", "GPT-2/3/4 · Llama · Qwen"),
    ("注意力", "全双向", "编码器双向 + 解码器因果 + 交叉", "因果（左到右）"),
    ("预训练目标", "MLM（填空）", "Span Corruption / 去噪", "自回归 LM（预测下一个 token）"),
    ("擅长", "分类/检索/抽取", "翻译/摘要/结构化生成", "开放生成/对话/推理"),
    ("参数效率", "高（理解任务）", "中", "生成任务上最优"),
    ("是否还能生成为主", "否", "可以但不主流", "是（2023 年后的绝对主流）"),
]
widths = [14, 26, 32, 34]
for r in ROWS:
    print("  " + "  ".join(str(c).ljust(w) for c, w in zip(r, widths)))

section("2) 时间线：从 2017 到今天")
TIMELINE = [
    ("2017", "Transformer", "提出注意力机制，机器翻译 SOTA"),
    ("2018", "BERT / GPT-1", "双向预训练 vs 自回归预训练，两条路线分道扬镳"),
    ("2019", "GPT-2 / XLNet / T5", "T5 用『everything is text-to-text』统一所有任务"),
    ("2020", "GPT-3 (175B)", "涌现能力 + 上下文学习（ICL），prompt 时代开启"),
    ("2022", "ChatGPT / InstructGPT", "RLHF 让模型『能对话』"),
    ("2023", "Llama / GPT-4 / Mistral", "开源追平闭源，Llama 生态爆炸；MoE 走向主流"),
    ("2024", "Llama-3 / Qwen2 / DeepSeek-V2/V3", "GQA/MLA + MoE + 长上下文成为标配"),
    ("2024+", "Mamba-2 / Jamba / 混合架构", "注意力与 SSM 混合，追求长序列效率"),
]
for year, name, desc in TIMELINE:
    print(f"  {year:<7}{name:<34}{desc}")

section("3) 主流开源模型速查表（2024~2025 常见配置）")
MODELS = [
    # 名称, 参数, 层数, d_model,  heads(kv), 上下文, 关键特性
    ("Llama-3-8B",     "8B",   32, 4096, "32/8(GQA)", "8k(128k)", "RoPE, SwiGLU, RMSNorm"),
    ("Llama-3-70B",    "70B",  80, 8192, "64/8(GQA)", "8k(128k)", "同上；训练 15T token"),
    ("Qwen2.5-7B",     "7B",   28, 3584, "28/4(GQA)", "32k(128k)", "中文强，词表 152k"),
    ("Mistral-7B",     "7B",   32, 4096, "32/8(GQA)", "32k",      "滑动窗口注意力 4096"),
    ("Mixtral-8x7B",   "47B(13B激活)", 32, 4096, "32/8(GQA)", "32k", "MoE: 8 专家选 2"),
    ("DeepSeek-V3",    "671B(37B激活)", 61, 7168, "128/MLA", "128k", "MLA + 256 专家 + 无辅助损失负载均衡"),
    ("GLM-4-9B",       "9B",   40, 4096, "32/2(GQA)", "128k",     "前缀式注意力变体 + 工具调用"),
    ("Mamba-2",        "2.7B", 64, 2560, "—",        "无限(递归)", "SSD：状态空间对偶性，O(T) 推理"),
]
print(f"  {'模型':<16}{'参数':<14}{'层':>4}{'d_model':>8}{'头数(Q/KV)':>14}{'上下文':>11}  特性")
for m in MODELS:
    print(f"  {m[0]:<16}{m[1]:<14}{m[2]:>4}{m[3]:>8}{m[4]:>14}{m[5]:>11}  {m[6]}")

section("4) 参数量速算公式")
def params_decoder_only(vocab, layers, d_model, d_ff=None, tie_embeddings=True):
    d_ff = d_ff or int(2.67 * d_model)          # SwiGLU 常用 8/3·d（保证参数量对齐）
    attn = 4 * d_model * d_model                # Q/K/V/O（GQA 会少一些）
    ffn = 3 * d_model * d_ff                    # SwiGLU 有三个矩阵
    emb = vocab * d_model
    head = 0 if tie_embeddings else vocab * d_model
    return layers * (attn + ffn) + emb + head


for name, v, l, d in [("Llama-3-8B", 128256, 32, 4096), ("Qwen2.5-7B", 152064, 28, 3584),
                      ("Llama-3-70B", 128256, 80, 8192)]:
    p = params_decoder_only(v, l, d)
    print(f"  {name:<14} 估算 = {human_num(p)}")
print("""
  误差来源：GQA 会减少 K/V 的参数，bias 项、额外 norm、MoE 的专家层都不在上面的公式里。
  但±10% 的估算足够做容量规划了。
""")

section("5) 选型建议（务实版）")
print("""
  中文任务 / 国内部署        → Qwen2.5 / GLM-4 / DeepSeek
  英文通用 + 生态最好        → Llama-3.1 系列（工具、量化、部署方案最全）
  显存紧张、要单卡跑         → Mistral-7B / Qwen2.5-7B + AWQ/GGUF 4-bit
  长文档（>64k）            → 原生长上下文版本（Qwen2.5-1M、Llama-3.1-128k、Gemini）
  高并发服务端               → MoE（DeepSeek / Mixtral）：同样算力下吞吐更高
  端侧 / CPU                 → 小模型 + GGUF（Qwen2.5-1.5B / Llama-3.2-1B）
  一句话：别迷信榜单，用你自己的数据做一个 50~200 条的评测集，跑一遍再决定。
""")
