"""
05 · 量化：把 16 位浮点压成 8 位 / 4 位整数
==============================================
为什么量化对 LLM 特别重要？
  · Decode 阶段是【访存受限】：每生成一个 token 都要把权重读一遍
  · 权重位宽减半 → 读的字节减半 → 速度几乎翻倍（这才是 4-bit 模型变快的真正原因）

主流方案：
  · 对称量化（absmax）: s = max|W| / 127,  W_int8 = round(W / s)
  · 非对称（零点）     : 处理分布偏移的情况
  · 分组量化 (GPTQ/AWQ): 每 32/128 个元素一组各自算 scale → 精度大幅提升
  · NF4 (QLoRA)       : 4-bit NormalFloat，16 个量化级别按正态分布密度布置

本脚本全部手写，并与 float 结果对比误差。
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import bytes_str, check_close, has_torch, section, set_seed, subsection

set_seed(19)

rng = np.random.default_rng(0)
M, K = 512, 256
W = rng.normal(0, 0.02, (M, K)).astype(np.float32)
# 混入几个"离群"权重，模拟真实 LLM 里少数异常大的元素（量化的头号敌人）
W[0, 0] = 0.85
W[7, 13] = -0.92
X = rng.normal(0, 1.0, (K, 128)).astype(np.float32)
ref = W @ X


# ----------------------------------------------------------------------------------
def absmax_quantize(w, bits=8, group_size=None):
    """对称 absmax 量化。group_size=None → 整张量一个 scale；否则按行分组。"""
    qmax = 2 ** (bits - 1) - 1
    if group_size is None:
        scale = np.abs(w).max() / qmax
        q = np.round(w / scale).clip(-qmax - 1, qmax)
        return q.astype(np.int32), np.array([scale])
    n_groups = w.shape[-1] // group_size
    w2 = w.reshape(-1, group_size)
    scales = np.abs(w2).max(axis=1, keepdims=True) / qmax
    q = np.round(w2 / scales).clip(-qmax - 1, qmax)
    return q.astype(np.int32).reshape(w.shape), scales.squeeze()


def dequantize(q, scales, group_size=None):
    if group_size is None:
        return q.astype(np.float32) * scales[0]
    return (q.reshape(-1, group_size).astype(np.float32) * scales[:, None]).reshape(q.shape)


def rel_error(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


section("1) per-tensor（整张量一个 scale）int8")
q8, s8 = absmax_quantize(W, bits=8)
W8 = dequantize(q8, s8)
print(f"  scale = {s8[0]:.6f}   (max|W| = {np.abs(W).max():.4f})")
print(f"  权重相对误差 = {rel_error(W8, W):.5f}")
print(f"  矩阵乘输出相对误差 = {rel_error(W8 @ X, ref):.5f}")
print(f"  压缩率 = {32 / 8:.0f}×（fp32 → int8），显存 {bytes_str(W.nbytes)} → {bytes_str(W.nbytes / 4)}")

section("2) 离群值的破坏力：两个 0.9 量级的元素逼着所有元素共享大 scale")
mask = np.zeros_like(W, dtype=bool)
mask[0, 0] = mask[7, 13] = True
print(f"  含离群值时的 scale = {s8[0]:.6f}")
clean = W.copy()
clean[mask] = 0
s_clean = np.abs(clean).max() / 127
print(f"  去掉离群值后的 scale = {s_clean:.6f}  → 小了 {s8[0] / s_clean:.1f}×")
print(f"  去掉离群值后的量化误差 = {rel_error(dequantize(*absmax_quantize(clean, 8)) @ X, clean @ X):.5f}")
print("  结论：LLM 里『超级权重 / 离群特征』是量化误差的主要来源，")
print("        AWQ / SmoothQuant 的核心就是把这些离群值『平滑』掉。")

section("3) 分组量化（group_size=64）：精度立刻回来")
for gs in [None, 128, 64, 32, 16]:
    q, sc = absmax_quantize(W, bits=8, group_size=gs)
    Wd = dequantize(q, sc, gs)
    extra = 0 if gs is None else (W.size // gs) * 4        # 每组一个 fp32 scale
    print(f"  group_size={str(gs):<6} 输出相对误差={rel_error(Wd @ X, ref):.6f}"
          f"   额外 scale 开销={bytes_str(extra)}")
print("  实践：GPTQ/AWQ 常用 group_size=128，几乎无损而开销可忽略。")

section("4) 4-bit NF4（QLoRA 用的那种）")
NF4_LEVELS = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022816467, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0])


def quantize_nf4(w, group_size=64):
    """把归一化后的值映射到最近的 NF4 级别（二分查找即可）。"""
    w2 = w.reshape(-1, group_size)
    scales = np.abs(w2).max(axis=1, keepdims=True)
    normed = w2 / (scales + 1e-12)
    idx = np.abs(normed[..., None] - NF4_LEVELS).argmin(axis=-1)
    return idx.astype(np.int32).reshape(w.shape), scales.squeeze()


def dequantize_nf4(idx, scales, group_size=64):
    v = NF4_LEVELS[idx.reshape(-1, group_size)]
    return (v * scales[:, None]).reshape(idx.shape)


idx4, sc4 = quantize_nf4(W, 64)
W4 = dequantize_nf4(idx4, sc4)
q8g, s8g = absmax_quantize(W, bits=8, group_size=64)
print(f"  4-bit NF4 (group=64) 权重相对误差 = {rel_error(W4, W):.5f}")
print(f"  int8     (group=64) 权重相对误差 = {rel_error(dequantize(q8g, s8g, 64), W):.5f}")
print(f"  int8     (per-tensor)权重相对误差 = {rel_error(W8, W):.5f}")
print(f"  4-bit 输出相对误差 = {rel_error(W4 @ X, ref):.5f}")
print(f"  显存: fp16 {bytes_str(W.nbytes / 2)} → int4 {bytes_str(W.nbytes / 8)}（再省一半）")
print("""
  QLoRA 的关键洞见：4-bit 量化权重 + fp16 的 LoRA 适配器，
  让 65B 模型能在一张 48G 卡上微调 —— 量化负责"存"，LoRA 负责"学"。
""")

section("5) 整数矩阵乘：真正的加速在哪里")
def int8_matmul(W_q, W_scale, X_q, X_scale):
    """int8 × int8 → int32 累加 → 乘回 scale（这就是 TensorCore INT8 干的事）"""
    out_i32 = W_q.astype(np.int64) @ X_q.astype(np.int64)
    return (out_i32 * W_scale * X_scale).astype(np.float32)


Xq, Xs = absmax_quantize(X, bits=8)
out_int8 = int8_matmul(q8, s8[0], Xq, Xs[0])
print(f"  int8 乘加输出相对误差 = {rel_error(out_int8, ref):.5f}")
print(f"  fp32 参考范数 = {np.linalg.norm(ref):.4f}")
print("""
  · W8A8（权重+激活都量化）：能用到 INT8 TensorCore，但激活有离群通道，难做
  · W4A16（只量化权重）    ：当前最实用（GPTQ/AWQ/GGUF 的 Q4_K_M 等），
    decode 时把权重反量化成 fp16 再算，省的是【显存带宽】而不是算力
  · KV Cache 也能量化（fp8 / int8），长上下文场景收益巨大
""")

section("6) 与 PyTorch 对照（可选）")
if has_torch:
    import torch

    Wt = torch.tensor(W)
    qt = torch.quantize_per_tensor(Wt, scale=float(s8[0]), zero_point=0, dtype=torch.qint8)
    check_close(torch.dequantize(qt).numpy(), W8, tol=1e-6, name="手写 absmax int8 vs torch.quantize_per_tensor")
else:
    print("[SKIP] 未安装 torch")

section("7) 选型速查")
print("""
  场景                       推荐
  --------------------------------------------------------------
  显存不够、要跑起来          GPTQ/AWQ 4-bit（W4A16）
  要最高质量                   fp16 / bf16 原权重
  要吞吐（服务端）            fp8（H100）+ W8A8 或 W4A16 + 连续批处理
  要微调（单卡）              QLoRA：4-bit 底座 + LoRA 适配器
  端侧 / CPU                  GGUF Q4_K_M / Q5_K_M（llama.cpp 生态）
  注意：量化后 perplexity 会上升，务必在自己的任务上做回归测试，
        尤其是数学/代码任务对量化更敏感。
""")
