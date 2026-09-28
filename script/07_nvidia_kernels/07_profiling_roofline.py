"""
07 · 性能分析：怎么判断"慢在哪"，以及 Roofline 模型
========================================================
没有测量就没有优化。GPU 上只需要回答两个问题：
  ① 这个算子是【算力受限】(compute-bound) 还是【访存受限】(memory-bound)？
  ② 距离硬件上限还差多少？（MFU：Model FLOPs Utilization）

Roofline 模型：
    可达性能 = min(峰值算力, 峰值带宽 × 算术强度)
    算术强度 AI = FLOPs / Bytes（每读 1 字节能算多少次）

工具链：
    nsys profile  → 时间线：哪些 kernel 在跑、有没有空隙、有没有同步等待
    ncu --set full → 单个 kernel 的详解：带宽、占用率、bank conflict、寄存器溢出
    torch.profiler → 从 Python 侧看整体（含 CPU 侧开销、kernel 名称）

本脚本：手测带宽与算力 → 画出 Roofline 的关键点 → 给出调优方向。
"""
import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np

from _cuda_helpers import cuda_available, has_torch
from common_utils import bytes_str, section, set_seed, subsection, timer

set_seed(7)
if not has_torch():
    print("[SKIP] 需要 PyTorch")
    raise SystemExit(0)
import torch

# ----------------------------------------------------------------------------------
section("1) 硬件上限（理论值）")
if cuda_available():
    p = torch.cuda.get_device_properties(0)
    bw_peak = p.memory_clock_rate * 2 * p.memory_bus_width / 8 / 1e9    # GB/s（DDR 双倍率）
    sm_count = p.multi_processor_count
    print(f"  GPU: {p.name}")
    print(f"    SM 数量          = {sm_count}")
    print(f"    显存带宽（理论） = {bw_peak:.0f} GB/s")
    print(f"    显存             = {p.total_memory / 1024**3:.1f} GB")
else:
    bw_peak, sm_count = 1555.0, 108          # 以 A100-80G 为参照
    print("  未检测到 GPU，下面用 A100-80G 的理论值作为参照：")
    print(f"    显存带宽（理论）= {bw_peak:.0f} GB/s, SM 数 = {sm_count}")

A100 = {
    "fp32 (非 TensorCore)": 19.5e12,
    "tf32 TensorCore": 156e12,
    "fp16/bf16 TensorCore": 312e12,
    "fp8 TensorCore (H100)": 1979e12,
}
print("\n  算力峰值（A100-80G 参考值，H100 约 2~3 倍）：")
for k, v in A100.items():
    print(f"    {k:<24}{v / 1e12:8.1f} TFLOPS")

# ----------------------------------------------------------------------------------
section("2) 实测带宽：一次纯访存操作能吃多少带宽")
if cuda_available():
    n = 1 << 26                                       # 256MB
    x = torch.empty(n, dtype=torch.float32, device="cuda")
    y = torch.empty_like(x)
    for _ in range(3):
        y.copy_(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        y.copy_(x)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 20 * 1000
    moved = 2 * n * 4 / 1e9
    bw = moved / (ms * 1e-3)
    print(f"  纯拷贝 {bytes_str(n * 4)} × 20 次：{ms:.3f} ms/次 → 实测带宽 {bw:.0f} GB/s "
          f"（理论 {bw_peak:.0f} GB/s，达成率 {bw / bw_peak:.0%}）")
    print("  经验：拷贝类操作通常能到理论带宽的 70%~90%，低于 60% 就要查是否没做向量化。")
else:
    # CPU 上用 numpy 测内存带宽，作为对照概念演示
    n = 1 << 24
    a = np.empty(n, dtype=np.float32)
    b = np.empty_like(a)
    t0 = time.perf_counter()
    for _ in range(5):
        b[:] = a
    dt = (time.perf_counter() - t0) / 5
    print(f"  CPU(numpy) 拷贝 {bytes_str(n * 4)}: {dt * 1000:.2f} ms → "
          f"{2 * n * 4 / dt / 1e9:.1f} GB/s（CPU 内存带宽通常 10~50 GB/s，比 GPU 低两个数量级）")

# ----------------------------------------------------------------------------------
section("3) 实测算力：GEMM 能到多少 TFLOPS")
if cuda_available():
    M = K = N = 4096
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    for _ in range(3):
        C = A @ B
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        C = A @ B
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 10 * 1000
    tflops = 2 * M * K * N / (ms * 1e-3) / 1e12
    print(f"  fp16 GEMM {M}×{K}×{N}: {ms:.3f} ms → {tflops:.1f} TFLOPS")
    print(f"  相对 fp16 TensorCore 峰值（312 TFLOPS）的利用率 = {tflops / 312:.1%}")
    print("""
  MFU（Model FLOPs Utilization）经验值：
    · 训练大模型：35%~50% 已经很好（其余花在通信、重计算、非 GEMM 算子）
    · 推理 prefill：40%~60%；decode：通常 <10%（因为 batch=1 时是访存受限）
""")
else:
    print("  [SKIP] 需要 GPU 才能测 GEMM 算力")

# ----------------------------------------------------------------------------------
section("4) Roofline：判断优化方向")
print(f"{'算子':<26}{'算术强度 (FLOP/Byte)':>24}{'判定':>18}")
OPS = [
    ("向量加法 / ReLU / GELU", 2 * 1 / (3 * 4)),          # 读2写1，每元素约1 FLOP
    ("RMSNorm / LayerNorm", 2 * 4 / (3 * 4)),
    ("softmax (行内, d=1024)", 5 * 1024 / (2 * 1024 * 4)),
    ("GEMM (batch=1, decode)", 2 * 4096 * 4096 / (4096 * 4096 * 2 * 2)),
    ("GEMM (M=N=K=4096)", 2 * 4096**3 / (3 * 4096**2 * 2)),
    ("注意力 (T=4096, d=128)", 2 * 2 * 4096**2 * 128 / (3 * 4096 * 128 * 2)),
]
ridge = 19.5e12 / (bw_peak * 1e9) if bw_peak else 12.5
for name, ai in OPS:
    verdict = "访存受限" if ai < ridge else "算力受限"
    print(f"  {name:<24}{ai:>24.2f}{verdict:>18}")
print(f"""
  分界点（ridge point）= 峰值算力 / 峰值带宽 = {ridge:.1f} FLOP/Byte
    AI < {ridge:.1f} → 访存受限：优化方向是【减少访存】（融合算子、向量化、用片上内存）
    AI > {ridge:.1f} → 算力受限：优化方向是【用更快的指令】（TensorCore、更低精度、减少非必要计算）
  这就是"为什么 decode 阶段量化能加速、而 prefill 阶段收益小"的定量解释：
    decode 是访存受限（读权重的字节数决定时间）→ 权重从 fp16 变 int4，时间直接减半
    prefill 是算力受限 → 位宽降低只减少访存，算力瓶颈不变
""")

# ----------------------------------------------------------------------------------
section("5) 工具怎么用")
print("""
  ▸ nsys（系统级时间线，看"哪里在空转"）
      nsys profile -o report python3 train.py
      nsys stats report.nsys-rep
      看什么：GPU 利用率曲线、kernel 之间的空隙（CPU 端没喂上数据？）、memcpy 占比

  ▸ ncu（kernel 级详解，看"这个 kernel 为什么慢"）
      ncu --set full -o prof python3 script/07_nvidia_kernels/05_matmul_tiled_run.py
      ncu --metrics sm__throughput.avg.pct_of_peak,dram__throughput.avg.pct_of_peak ./matmul
      看什么：
        Compute (SM) Throughput  vs  Memory Throughput   → 判断 compute/memory bound
        Achieved Occupancy                               → 并行度够不够
        Registers Per Thread / Local Memory              → 有没有寄存器溢出(spill)
        Shared Memory Bank Conflicts                     → shared 访问模式有没有问题

  ▸ torch.profiler（Python 侧，最省事）
      with torch.profiler.profile(activities=[ProfilerActivity.CUDA]) as prof:
          ...
      print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
""")
if cuda_available():
    from torch.profiler import ProfilerActivity, profile

    A = torch.randn(1024, 1024, device="cuda")
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(20):
            C = A @ A
            C = torch.softmax(C, dim=-1)
        torch.cuda.synchronize()
    print("  torch.profiler 输出（前 6 行）：")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=6))

section("6) 一张调优清单")
print("""
  ① 先确认是 compute-bound 还是 memory-bound（ncu 一眼可见）
  ② memory-bound：融合算子 / float4 向量化 / 减少中间结果落盘 / 用 shared memory 复用
  ③ compute-bound：上 TensorCore（fp16/bf16/fp8）/ 减少分支 / 提高 tile 与 ILP
  ④ occupancy 低：减少寄存器用量（拆分 kernel 或 -maxrregcount）、减小 shared 占用
  ⑤ launch 开销大（kernel 太小）：合并成更大的 kernel，或用 CUDA Graph 消除启动开销
  ⑥ 端到端还是慢：看是不是 CPU 端瓶颈（tokenizer、调度、数据拷贝），
     用 CUDA Graph / 连续批处理 / 多 stream 解决
""")
