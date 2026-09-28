"""
09 · 大模型是怎么训出来的：并行策略与混合精度
================================================
单卡装不下、单卡算不完 → 必须切分。五种并行 + 两种省显存技术：

  DP (Data Parallel)        每张卡拿一份完整模型、不同数据 → 梯度 all-reduce
  ZeRO-1/2/3                DP 的进化：把优化器状态 / 梯度 / 参数也切分到各卡
  TP (Tensor Parallel)      把单个矩阵乘法按行/列切开（Megatron-LM）
  PP (Pipeline Parallel)    把不同层放到不同卡上，像流水线一样传递激活
  SP (Sequence Parallel)    把序列维度切开（长上下文必备，配合 TP 使用）
  EP (Expert Parallel)      MoE 专用：不同专家放不同卡

外加：
  混合精度（fp16/bf16 + fp32 主权重 + loss scaling）
  激活重计算（gradient checkpointing）
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common_utils import bytes_str, human_num, section, subsection

section("1) 显存账本：7B 模型混合精度训练")
P = 7e9
items = [
    ("模型权重 (bf16)", 2 * P),
    ("梯度 (bf16)", 2 * P),
    ("fp32 主权重", 4 * P),
    ("Adam m (fp32)", 4 * P),
    ("Adam v (fp32)", 4 * P),
    ("激活 (batch=8, seq=2048, 开重计算)", 8 * 2048 * 4096 * 32 * 2 * 2),
]
total = sum(x for _, x in items)
print(f"  {'项目':<40}{'显存':>12}{'占比':>8}")
for name, b in items:
    print(f"  {name:<40}{bytes_str(b):>12}{b / total:>8.1%}")
print(f"  {'合计':<40}{bytes_str(total):>12}")
print("""
  结论：单卡 80G 装不下 7B 的训练状态（约 112GB）→ 必须 ZeRO 或 TP/PP
  推理时只需"权重"这一项（14GB）→ 所以 A100 40G 能跑 7B 推理，却训不了
""")

section("2) 五种并行怎么选")
print(f"""
  {'策略':<8}{'切什么':<28}{'通信量':<20}{'适用场景'}
  {'-' * 88}
  DP     {'数据（模型各卡一份）':<28}{'梯度 all-reduce':<20}{'模型能装进单卡'}
  ZeRO-1 {'优化器状态':<28}{'reduce-scatter':<20}{'想省优化器显存'}
  ZeRO-2 {'+ 梯度':<28}{'reduce-scatter':<20}{'再省一层'}
  ZeRO-3 {'+ 参数（按需 gather）':<28}{'all-gather（频繁）':<20}{'模型装不进单卡'}
  TP     {'单层的矩阵（列/行切）':<28}{'all-reduce（每层 2 次）':<20}{'单卡装不下单层，需 NVLink'}
  PP     {'层（不同层放不同卡）':<28}{'只传边界激活':<20}{'层数多、跨机'}
  SP     {'序列维度':<28}{'all-gather / reduce-scatter':<20}{'长上下文 + TP 组合'}
  EP     {'MoE 专家':<28}{'all-to-all':<20}{'MoE 模型'}
""")
print("""  实践组合（3D 并行）：TP 在机内（NVLink 快），PP 跨机，DP/ZeRO 兜底
  经验值：
    · 单机 8 卡：优先 ZeRO-2/3（改动最小）
    · TP 一般取 2/4/8，且必须能被 head 数整除；跨机不要开 TP（通信太慢）
    · PP 会引入 bubble（空转），用 1F1B 调度 + 更小的 micro-batch 缓解
""")

section("3) 混合精度：为什么大模型都用 bf16 而不是 fp16")
print("""
           指数位   尾数位   动态范围        是否需要 loss scaling
  fp32     8        23      极大           不需要
  fp16     5        10      小（易溢出）    需要（否则梯度下溢）
  bf16     8        7       同 fp32        不需要

  bf16 的尾数位只有 7 位（精度很低），但动态范围和 fp32 一样 → 不容易溢出/下溢
  → 训练稳定性大幅提升，所以 2023 年后几乎全用 bf16（A100/H100 原生支持）
  注意：bf16 的精度损失会让"小更新"被吞掉 → 所以必须保留 fp32 主权重做累加
""")

section("4) 激活重计算：用算力换显存")
B, T, D, L = 8, 2048, 4096, 32
act_full = B * T * D * L * 2                    # fp16，粗略
act_ckpt = B * T * D * (L ** 0.5) * 2           # 只保存每 sqrt(L) 层的输入
print(f"  不重计算: {bytes_str(act_full)}")
print(f"  重计算  : {bytes_str(act_ckpt)}（只存检查点，反向时重算中间层）")
print(f"  代价：多算一次前向 ≈ +30% 时间，换来 {(act_full / act_ckpt):.0f}× 激活显存下降")
print("""
  几乎所有大模型训练都会开（选择性重计算：只重算注意力部分，FFN 不重算）
""")

section("5) 一个 70B 训练的配置示例")
print("""
  硬件：8 机 × 8 卡 A100-80G（64 卡）
  并行：TP=4（机内） × PP=4（跨机） × DP=4，ZeRO-1 叠加
  精度：bf16 + fp32 主权重，flash-attn，gradient checkpointing
  批量：global batch = 4M token（micro-batch 8 × seq 4096 × 累积若干步）
  优化：AdamW（β=0.9/0.95），lr=1.5e-4，warmup 2000 步，余弦衰减到 10%
  稳定性：梯度裁剪 1.0；loss spike 时回滚到上一个 checkpoint 并重跑该段数据
""")

section("6) 训练不稳定的常见原因（血泪经验）")
print("""
  ① loss 尖峰：多由"坏数据"（超长重复文本、乱码）引起 → 数据清洗 + 梯度裁剪 + 回滚
  ② bf16 下溢：某些层的梯度一直是 0 → 检查是否被 cast 成 fp16、是否忘了 loss scaling
  ③ 并行引入的 bug：TP 的 RowParallel 输出需要 all-reduce，漏了会静默错
  ④ 位置编码越界：序列超过 max_position_embeddings → 静默输出乱码
  ⑤ 权重初始化过大 → 一开始就 NaN；现代做法是"小初始化 + Pre-LN/RMSNorm"
""")
