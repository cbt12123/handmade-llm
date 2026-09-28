"""
06 · 用 Triton 写算子：比 CUDA 好写 10 倍，性能接近手写
==========================================================
Triton 是什么：
  · 用 Python 语法写 GPU kernel，编译器帮你做【寄存器分配、shared memory 管理、指令调度】
  · 你只需要关心"每个 program 处理哪块数据"（tile 级别），不用管 warp/thread
  · OpenAI / PyTorch 生态大量使用；torch.compile 的后端之一就是 Triton

对照：
  CUDA   : 你要管理 threadIdx / blockIdx / __syncthreads / shared memory
  Triton : 你写 `pid = program_id(0)` + `tl.load(ptr + offsets)`，其余交给编译器

本脚本实现三个 kernel：elementwise GELU、softmax、分块 matmul。
"""
import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np

from _cuda_helpers import cuda_available, has_torch, has_triton, skip
from common_utils import check_close, section, set_seed, subsection

set_seed(6)
if not has_torch():
    skip("需要 PyTorch：pip install torch")
import torch
import torch.nn.functional as F

section("1) Triton 编程模型：一张对照表")
print("""
  CUDA 概念              Triton 对应                        说明
  -----------------------------------------------------------------------------
  blockIdx               program_id(axis)                  每个 program 处理一个 tile
  blockDim               num_warps=4（编译器决定线程数）     以 warp 为单位，通常 4 或 8
  threadIdx              （隐藏）tl.arange(0, BLOCK)         用"向量化索引"代替线程索引
  __syncthreads()        （不需要）                         编译器保证依赖
  shared memory          （不需要显式管理）                   编译器自动分配
  if (i < n) 边界检查    mask= 参数传给 tl.load / tl.store   用掩码代替分支
  grid 配置              kernel[(grid,)](...)                一维/二维/三维 grid
""")

if not cuda_available():
    skip("没有可用的 CUDA GPU，无法运行 Triton kernel。\n"
         "        CPU 环境可以先读下面的源码理解编程模型，再到容器里跑。")
if not has_triton():
    skip("未安装 triton：pip install triton（torch 2.x 通常已自带）")

import triton
import triton.language as tl

# ----------------------------------------------------------------------------------
section("2) Triton kernel ①：elementwise GELU")
@triton.jit
def gelu_kernel(X, Y, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(X + offsets, mask=mask, other=0.0)
    # tanh 近似的 GELU
    y = 0.5 * x * (1.0 + tl.math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    tl.store(Y + offsets, y, mask=mask)


def gelu_triton(x: torch.Tensor):
    y = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    gelu_kernel[grid](x, y, n, BLOCK_SIZE=BLOCK, num_warps=4)
    return y


x = torch.randn(1 << 20, device="cuda")
y_tri = gelu_triton(x)
check_close(y_tri, F.gelu(x, approximate="tanh"), tol=1e-6, name="Triton GELU vs torch")

# ----------------------------------------------------------------------------------
section("3) Triton kernel ②：行 softmax")
@triton.jit
def softmax_kernel(X, Y, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(X + row * n_cols + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)                      # 数值稳定
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    tl.store(Y + row * n_cols + cols, num / den, mask=mask)


def softmax_triton(x: torch.Tensor):
    y = torch.empty_like(x)
    rows, cols = x.shape
    BLOCK = triton.next_power_of_2(cols)
    softmax_kernel[(rows,)](x, y, cols, BLOCK=BLOCK, num_warps=4)
    return y


xs = torch.randn(128, 513, device="cuda")          # 故意用非 2 的幂，考验 mask
check_close(softmax_triton(xs), F.softmax(xs, dim=-1), tol=1e-6, name="Triton softmax vs torch")
print("  非 2 的幂的列数（513）也能正确处理 —— mask 参数就是干这个的。")

# ----------------------------------------------------------------------------------
section("4) Triton kernel ③：分块矩阵乘法")
@triton.jit
def matmul_kernel(A, B, C, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(A + offs_m[:, None] * K + (k + offs_k)[None, :],
                    mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K), other=0.0)
        b = tl.load(B + (k + offs_k)[:, None] * N + offs_n[None, :],
                    mask=((k + offs_k)[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)                        # ← 这一行会被编译成 TensorCore 指令
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul_triton(a: torch.Tensor, b: torch.Tensor, bm=64, bn=64, bk=32):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))
    matmul_kernel[grid](a, b, c, M, N, K,
                        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=4, num_stages=3)
    return c


A = torch.randn(512, 1024, device="cuda")
B = torch.randn(1024, 768, device="cuda")
C_tri = matmul_triton(A, B)
check_close(C_tri, A @ B, tol=1e-2, name="Triton matmul vs torch.matmul")

section("5) 简单 autotune：让编译器帮你选 tile 大小")
@triton.autotune(
    configs=[triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
             triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
             triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=4)],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_autotune(A, B, C, M, N, K,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(A + offs_m[:, None] * K + (k + offs_k)[None, :], other=0.0)
        b = tl.load(B + (k + offs_k)[:, None] * N + offs_n[None, :], other=0.0)
        acc += tl.dot(a, b)
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc)


def bench(fn, iters=50):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


M, K, N = 1024, 1024, 1024
A2 = torch.randn(M, K, device="cuda")
B2 = torch.randn(K, N, device="cuda")
ms_tri = bench(lambda: matmul_triton(A2, B2))
ms_torch = bench(lambda: A2 @ B2)
flops = 2 * M * K * N
print(f"  {M}×{K}×{N} GEMM：")
print(f"    Triton (BLOCK=64x64x32) : {ms_tri:7.3f} ms  {flops / (ms_tri * 1e-3) / 1e9:8.1f} GFLOPS")
print(f"    torch.matmul (cuBLAS)   : {ms_torch:7.3f} ms  {flops / (ms_torch * 1e-3) / 1e9:8.1f} GFLOPS")
print("""
  不开 autotune、不调优的 Triton 版本通常能达到 cuBLAS 的 60%~90%，
  加上 autotune + num_stages 调优后可以接近甚至超过（特定形状下）。
  这就是 Triton 的价值：用 1/10 的代码量换 80%~100% 的性能。
""")

section("6) Triton 的常用技巧")
print("""
  ① num_warps=4/8：影响每个 program 用多少 warp；tile 越大越要多 warp
  ② num_stages=2~5：软件流水线级数（预取多少块到 shared），越大越吃 shared memory
  ③ tl.dot(a, b) 会自动用 TensorCore（fp16/bf16 输入时；fp32 会用 TF32 或 FMA）
  ④ 用 tl.load(..., mask=..., other=...) 处理边界，避免分支
  ⑤ 用 tl.where 代替 if；用 tl.math.exp2 代替 exp（更快：2^x 有硬件指令）
  ⑥ 调试：设置 TRITON_INTERPRET=1 可以在 CPU 上逐行调试（很慢但能看到中间值）
  ⑦ 与 torch.compile 配合：torch.compile(model) 会自动把逐元素算子融合成 Triton kernel
""")
