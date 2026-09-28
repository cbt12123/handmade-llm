"""
05 · 编译并运行 tiled GEMM（无 GPU 时用 numpy 定量演示"分块为什么快"）
=========================================================================
核心问题：为什么把数据搬进 shared memory 就能快十几倍？
  答：不是 shared memory 有多快，而是【全局访存次数少了 TILE 倍】

  V1 朴素：C 的每个元素都要独立读 A 的一行(K) 和 B 的一列(K)
          → 全局访存 ≈ 2·M·N·K
  V2 分块：一个 16×16 的 tile 搬进 shared 后被 256 个线程复用
          → 全局访存 ≈ 2·M·N·K / TILE

本脚本用 numpy 精确统计两种写法的访存次数（不依赖 GPU），并在有 GPU 时真跑 CUDA。
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np

from _cuda_helpers import compile_cu, ensure_dir, nvcc_available
from common_utils import section, subsection, timer

HERE = Path(__file__).resolve().parent

section("1) 用 numpy 统计访存次数（算术强度分析）")
M = N = K = 256
TILE = 16
naive_reads = 2 * M * N * K                       # 每个输出元素读 K 个 A + K 个 B
tiled_reads = M * K * (N // TILE) + K * N * (M // TILE)
print(f"  矩阵规模: {M}×{K} × {K}×{N}")
print(f"  V1 朴素 全局读 = {naive_reads:,} 个 float ({naive_reads * 4 / 1e6:.1f} MB)")
print(f"  V2 分块 全局读 = {tiled_reads:,} 个 float ({tiled_reads * 4 / 1e6:.1f} MB)")
print(f"  访存减少 {naive_reads / tiled_reads:.0f}×  → 这正是分块最大的收益来源")
print(f"  计算量不变 = {2 * M * N * K:,} FLOPs")
print(f"  算术强度(AI): 朴素 = {2 * M * N * K / (naive_reads * 4):.2f} FLOP/Byte, "
      f"分块 = {2 * M * N * K / (tiled_reads * 4):.2f} FLOP/Byte")
print("""
  Roofline 视角：AI 越高，越可能达到"算力上限"而不是被带宽卡死
  A100: 峰值算力 312 TFLOPS(fp32/19.5 TFLOPS fp32, 实际 fp32 非tensor 约 19.5)、带宽 1555 GB/s
  要让 GEMM 跑满算力，需要 AI > 19.5e12 / 1555e9 ≈ 12.5 FLOP/Byte
  → 所以 GEMM 必须分块（把 AI 从 0.5 提到 8+），再配合更大的 tile 与寄存器分块才能达标
""")

section("2) 真的用 numpy 跑一遍（验证分块结果一致）")
rng = np.random.default_rng(0)
A = rng.normal(size=(M, K)).astype(np.float32)
B = rng.normal(size=(K, N)).astype(np.float32)
with timer("numpy 直接 matmul"):
    C_ref = A @ B


def matmul_tiled_numpy(A, B, tile=16):
    """显式模拟分块：把 tile 取出来做小矩阵乘，累加到输出块。"""
    m, k = A.shape
    _, n = B.shape
    C = np.zeros((m, n), dtype=np.float32)
    for i in range(0, m, tile):
        for j in range(0, n, tile):
            acc = np.zeros((min(tile, m - i), min(tile, n - j)), dtype=np.float32)
            for t in range(0, k, tile):
                acc += A[i:i + tile, t:t + tile] @ B[t:t + tile, j:j + tile]
            C[i:i + tile, j:j + tile] = acc
    return C


with timer("numpy 分块 matmul（Python 循环版）"):
    C_tiled = matmul_tiled_numpy(A, B, TILE)
print(f"  两种写法的最大误差 = {np.abs(C_ref - C_tiled).max():.3e}")
print("  注意：Python 循环版更慢（解释器开销），GPU 上分块才真正更快 —— 因为省的是访存。")

section("3) 编译并运行 CUDA 版本")
if not nvcc_available():
    print("[SKIP] 未找到 nvcc。在 docker/Dockerfile 容器里运行：")
    print("       python3 script/07_nvidia_kernels/05_matmul_tiled_run.py")
else:
    build = ensure_dir()
    exe = compile_cu(HERE / "05_matmul_tiled.cu", build / "matmul", libs=["cublas"])
    from _cuda_helpers import run_binary

    print(run_binary(exe))

section("4) 生产里该用什么")
print("""
  自己写 GEMM 的收益极低（cuBLAS/CUTLASS 已经把能做的都做了），
  但要懂原理，因为你要能：
    ① 读懂 ncu 报告里的 "memory throughput / compute throughput"
    ② 判断一个算子是 compute-bound 还是 memory-bound（决定优化方向）
    ③ 写"融合算子"（GEMM + bias + GELU、GEMM + softmax）时知道怎么切块
  真正值得自己写的通常是：
    · 框架里没有的融合算子（如 RMSNorm+Residual、RoPE、MoE 的 gather/scatter）
    · 特定形状下的专用算子（小 batch、超长序列、稀疏/变长）
    · 量化算子（fp8/int4 的 dequant-GEMM 融合）
  工具选择优先级：Triton > CUTLASS > 手写 CUDA（开发效率 vs 极致性能）
""")
