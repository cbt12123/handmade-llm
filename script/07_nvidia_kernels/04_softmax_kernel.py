"""
04 · Softmax：数值稳定 + 在线 softmax（FlashAttention 的前置知识）
==================================================================
    softmax(x_i) = exp(x_i − max(x)) / Σ exp(x_j − max(x))

为什么必须减 max：exp(1000) 会直接 inf → nan
在线 softmax（online softmax）：流式处理时维护 (m, l, O) 三元组
    新来一块数据：
        m_new = max(m, max(block))
        O     = O · exp(m − m_new) + exp(block − m_new) · V
        l     = l · exp(m − m_new) + Σ exp(block − m_new)
    → 这就是 FlashAttention 能在不保存 T×T 矩阵的情况下得到精确结果的原因

本脚本：
  1. 手写行 softmax kernel（每 block 一行，warp reduce）
  2. 手写在线 softmax（分块流式），验证与一次性 softmax 完全等价
"""
import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np

from _cuda_helpers import cuda_available, has_torch, skip
from common_utils import check_close, section, set_seed, subsection

set_seed(4)
if not has_torch():
    skip("需要 PyTorch：pip install torch")
import torch
import torch.nn.functional as F

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__device__ __forceinline__ float warp_max(float v) {
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

// ---------------------------------------------------------------------------
// 行 softmax：一个 block 处理一行，三趟扫描（max → exp&sum → 归一化）
// ---------------------------------------------------------------------------
__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int d) {
    int row = blockIdx.x;
    const float* xr = x + (size_t)row * d;
    float* yr = y + (size_t)row * d;

    extern __shared__ float smem[];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;

    // ---- 第一趟：求最大值 ----
    float m = -FLT_MAX;
    for (int i = threadIdx.x; i < d; i += blockDim.x) m = fmaxf(m, xr[i]);
    m = warp_max(m);
    if (lane == 0) smem[wid] = m;
    __syncthreads();
    m = (threadIdx.x < blockDim.x / 32) ? smem[lane] : -FLT_MAX;
    if (wid == 0) m = warp_max(m);

    // ---- 第二趟：exp 与求和 ----
    float s = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) s += __expf(xr[i] - m);
    s = warp_sum(s);
    if (lane == 0) smem[wid] = s;
    __syncthreads();
    s = (threadIdx.x < blockDim.x / 32) ? smem[lane] : 0.0f;
    if (wid == 0) s = warp_sum(s);

    // ---- 第三趟：写回 ----
    float inv = 1.0f / s;
    for (int i = threadIdx.x; i < d; i += blockDim.x) yr[i] = __expf(xr[i] - m) * inv;
}

// ---------------------------------------------------------------------------
// 在线 softmax 的演示版：每个线程独立跑一遍"流式处理"，验证与一次性结果一致
// （真实 FlashAttention 是 block 内共享 (m,l) 的，这里为了教学拆到线程级）
// ---------------------------------------------------------------------------
__global__ void online_softmax_kernel(const float* __restrict__ x, float* __restrict__ y,
                                      int d, int block_len) {
    int row = blockIdx.x;
    const float* xr = x + (size_t)row * d;
    float* yr = y + (size_t)row * d;
    if (threadIdx.x != 0) return;                  // 单线程演示算法流程

    float m = -FLT_MAX, l = 0.0f;
    // 第一趟：在线维护 m 和 l（不保存任何中间矩阵）
    for (int b = 0; b < d; b += block_len) {
        int end = min(b + block_len, d);
        float bm = -FLT_MAX;
        for (int i = b; i < end; ++i) bm = fmaxf(bm, xr[i]);
        float m_new = fmaxf(m, bm);
        float corr = __expf(m - m_new);             // 修正因子
        float bs = 0.0f;
        for (int i = b; i < end; ++i) bs += __expf(xr[i] - m_new);
        l = l * corr + bs;                          // 累积分母
        m = m_new;
    }
    // 第二趟：用最终的 m、l 写回（真实实现里这两趟是合并的）
    for (int i = 0; i < d; ++i) yr[i] = __expf(xr[i] - m) / l;
}

torch::Tensor softmax_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2, "x: (rows, d) CUDA tensor");
    int rows = x.size(0), d = x.size(1);
    auto y = torch::empty_like(x);
    int threads = 256;
    while (threads > 32 && threads > d) threads >>= 1;
    softmax_kernel<<<rows, threads, (threads / 32) * sizeof(float)>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), d);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor online_softmax_cuda(torch::Tensor x, int block_len) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2, "x: (rows, d) CUDA tensor");
    auto y = torch::empty_like(x);
    online_softmax_kernel<<<x.size(0), 32>>>(x.data_ptr<float>(), y.data_ptr<float>(),
                                             x.size(1), block_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
"""

CPP_SRC = """
torch::Tensor softmax_cuda(torch::Tensor x);
torch::Tensor online_softmax_cuda(torch::Tensor x, int64_t block_len);
"""

section("1) 数值稳定性：不减 max 会怎样")
x_big = torch.tensor([1000.0, 1001.0, 1002.0])
print(f"  直接 exp: {torch.exp(x_big).tolist()}   ← inf → softmax 变 nan")
x_safe = torch.exp(x_big - x_big.max())
print(f"  减 max 后: {(x_safe / x_safe.sum()).tolist()}")
print(f"  torch.softmax: {F.softmax(x_big, dim=-1).tolist()}")

section("2) 在线 softmax 的 CPU 演示（FlashAttention 的核心算法）")


def online_softmax_numpy(x, block_len=4):
    """单趟流式：只维护 m（running max）和 l（running sum），内存 O(1)。"""
    m, l = -np.inf, 0.0
    for b in range(0, len(x), block_len):
        blk = x[b:b + block_len]
        bm = blk.max()
        m_new = max(m, bm)
        corr = np.exp(m - m_new) if np.isfinite(m) else 0.0
        l = l * corr + np.exp(blk - m_new).sum()
        m = m_new
    return np.exp(x - m) / l


x_np = np.random.default_rng(0).normal(size=16) * 3
ref = np.exp(x_np - x_np.max()) / np.exp(x_np - x_np.max()).sum()
online = online_softmax_numpy(x_np, block_len=4)
print(f"  一次性 softmax 与在线 softmax 的最大差 = {np.abs(ref - online).max():.3e}")
print("  完全等价 → 所以 FlashAttention 不需要保存完整的 T×T 矩阵也能得到精确结果。")

section("3) 编译 CUDA 版")
if not cuda_available():
    skip("没有可用的 CUDA GPU，跳过编译（上面第 2 节已用 numpy 验证了算法等价性）")
    mod = None
else:
    from torch.utils.cpp_extension import load_inline

    try:
        mod = load_inline(name="softmax_ext", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC,
                          functions=["softmax_cuda", "online_softmax_cuda"],
                          extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)
        print("  编译成功")
    except Exception as e:
        print(f"  编译失败: {str(e)[:200]}")
        mod = None

section("4) 正确性与性能")
if mod is not None:
    xg = torch.randn(256, 1024, device="cuda")
    y = mod.softmax_cuda(xg)
    check_close(y, F.softmax(xg, dim=-1), tol=1e-6, name="自定义 softmax vs torch.softmax")
    yo = mod.online_softmax_cuda(xg, 64)
    check_close(yo, F.softmax(xg, dim=-1), tol=1e-5, name="在线 softmax vs torch.softmax")

    def bench(fn, iters=200):
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000

    big = torch.randn(8192, 1024, device="cuda")
    moved = 2 * big.numel() * 4 / 1e6
    for name, fn in [("torch.softmax", lambda: F.softmax(big, dim=-1)),
                     ("自定义 softmax_cuda", lambda: mod.softmax_cuda(big))]:
        ms = bench(fn)
        print(f"    {name:<24}{ms:8.3f} ms   带宽 {moved / ms / 1e3:7.1f} GB/s")

section("5) 调优要点")
print("""
  ① 每行一个 block 只在 d 不太大时高效；d 很大（>2048）时改用"每 block 多行"或 warp-per-row
  ② d ≤ 32 时可以用【一个 warp 处理一行】，完全不需要 __syncthreads
  ③ shared memory 的 bank conflict：
     32 个 bank，若同一 warp 的 32 个线程访问同一 bank 的不同地址 → 串行化
     规避：把数组大小设成 33（padding）或改用 xor 索引（ butterfly 归约天然无冲突）
  ④ __expf 是快速近似（约 2 ulp），比 expf 快很多；--use_fast_math 会全局开启
  ⑤ 在线 softmax 要注意：修正因子 exp(m_old − m_new) 在 m 未更新时是 exp(-inf)=0，
     所以第一次迭代要特殊处理（或用 m 初始化为 -FLT_MAX 并让 corr 自然为 0）
""")
