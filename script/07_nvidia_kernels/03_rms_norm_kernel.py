"""
03 · 融合算子实战：RMSNorm（LLM 里每层都要跑两次的那个）
==========================================================
    y = x / sqrt(mean(x²) + eps) * w

朴素写法要三个 kernel：x² → 求和 → 除法乘 w（三次全局访存）
融合成一个 kernel：每个 block 负责一行，用【shared memory + warp shuffle】做归约
    → 全局访存从 3 次降到 1 次，这就是"融合（fusion）"的价值

顺带讲清 CUDA 归约的两个核心原语：
  ① __shfl_down_sync：warp 内寄存器级交换数据（不需要 shared memory）
  ② __syncthreads() + shared memory：跨 warp 的归约
"""
import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np

from _cuda_helpers import cuda_available, has_torch, skip
from common_utils import check_close, section, set_seed, subsection

set_seed(3)
if not has_torch():
    skip("需要 PyTorch：pip install torch")
import torch
import torch.nn as nn

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// warp 级归约：用 __shfl_xor_sync 做"蝴蝶交换"，5 步把 32 个数加成一个
// 每个线程都能拿到总和（不需要广播），且完全在寄存器里完成 —— 最快
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    return v;
}

// ---------------------------------------------------------------------------
// block 级归约：先 warp 内归约，再由 0 号 lane 写入 shared，
// 最后前 32 个线程做一次 warp 归约
// ---------------------------------------------------------------------------
__device__ __forceinline__ float block_sum(float v, float* shared) {
    int lane = threadIdx.x & 31;          // threadIdx.x % 32
    int wid  = threadIdx.x >> 5;          // threadIdx.x / 32
    v = warp_sum(v);                      // 每个 warp 得到自己那部分的和
    if (lane == 0) shared[wid] = v;       // 各 warp 的结果写入 shared
    __syncthreads();                      // 必须同步：否则可能读到别的 warp 还没写的值
    // 用前 32 个线程对 shared 里的 warp 结果再归约一次
    v = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) v = warp_sum(v);
    return v;
}

// ---------------------------------------------------------------------------
// 融合 RMSNorm：一个 block 处理一行
//   grid.x  = 行数（每行一个 block）
//   block.x = threads（通常 256，需 >= 32 且能覆盖一行）
// ---------------------------------------------------------------------------
__global__ void rms_norm_kernel(const float* __restrict__ x,
                                const float* __restrict__ w,
                                float* __restrict__ y,
                                int d, float eps) {
    int row = blockIdx.x;                       // 一行一个 block
    const float* xr = x + (size_t)row * d;
    float* yr = y + (size_t)row * d;

    extern __shared__ float shared[];           // 动态 shared memory：存放各 warp 的部分和

    // ---- 第一遍：算平方和（每个线程累加自己 stride 步长的元素）----
    float sum = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    sum = block_sum(sum, shared);               // 归约得到整行的平方和

    // ---- 计算缩放系数（只算一次，然后所有线程复用）----
    float inv = rsqrtf(sum / (float)d + eps);   // rsqrtf 是硬件指令，比 1/sqrtf 快

    // ---- 第二遍：写回结果（注意：x 被读了两次，但都在 L1/L2 里命中）----
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        yr[i] = xr[i] * inv * w[i];
    }
}

torch::Tensor rms_norm_cuda(torch::Tensor x, torch::Tensor w, float eps) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2, "x: (rows, d) CUDA tensor");
    int rows = x.size(0);
    int d = x.size(1);
    auto y = torch::empty_like(x);

    int threads = 256;
    while (threads > 32 && threads > d) threads >>= 1;   // 线程数不必超过维度
    size_t shared_bytes = (threads / 32) * sizeof(float);
    rms_norm_kernel<<<rows, threads, shared_bytes>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(), d, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
"""

CPP_SRC = "torch::Tensor rms_norm_cuda(torch::Tensor x, torch::Tensor w, double eps);"

section("1) 先写 CPU/torch 参考实现")


def rms_norm_ref(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
    """PyTorch 版（与 nn.RMSNorm 一致）"""
    var = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * w


M, D = 64, 1024
x = torch.randn(M, D)
w = torch.randn(D) * 0.02 + 1.0
ref = rms_norm_ref(x, w)
print(f"  输入形状 {tuple(x.shape)}，eps=1e-6")
print(f"  参考实现输出: 均值={ref.mean():.4f} 标准差={ref.std():.4f}")
try:
    torch_ref = nn.RMSNorm(D)(x)
    print(f"  torch.nn.RMSNorm 输出形状 {tuple(torch_ref.shape)}（权重初始化为 1，数值上应与上面一致）")
except AttributeError:
    print("  （当前 torch 版本没有 nn.RMSNorm，用上面的参考实现即可）")

section("2) 编译 CUDA 版")
if not cuda_available():
    skip("没有可用的 CUDA GPU，跳过编译。CPU 参考实现已验证，逻辑见源码注释。")
    mod = None
else:
    from torch.utils.cpp_extension import load_inline

    try:
        mod = load_inline(name="rms_norm_ext", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC,
                          functions=["rms_norm_cuda"], extra_cuda_cflags=["-O3"], verbose=False)
        print("  编译成功")
    except Exception as e:
        print(f"  编译失败: {str(e)[:200]}")
        mod = None

section("3) 正确性与性能")
if mod is not None:
    xg, wg = x.cuda(), w.cuda()
    y = mod.rms_norm_cuda(xg, wg, 1e-6)
    check_close(y, rms_norm_ref(xg, wg), tol=1e-5, name="自定义 RMSNorm vs torch 参考实现")

    def bench(fn, iters=200):
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000

    big = torch.randn(4096, 4096, device="cuda")
    wbig = torch.randn(4096, device="cuda") * 0.02 + 1.0
    ms_cuda = bench(lambda: mod.rms_norm_cuda(big, wbig, 1e-6))
    ms_torch = bench(lambda: rms_norm_ref(big, wbig))
    moved = 2 * big.numel() * 4 / 1e6
    print(f"  数据: {tuple(big.shape)}（{moved:.0f} MB，读+写）")
    print(f"    自定义 CUDA kernel : {ms_cuda:8.3f} ms   带宽 {moved / ms_cuda / 1e3:7.1f} GB/s")
    print(f"    torch 组合实现     : {ms_torch:8.3f} ms   带宽 {moved / ms_torch / 1e3:7.1f} GB/s")
    print(f"    加速比: {ms_torch / ms_cuda:.2f}×")
    print("""
  torch 的组合实现会生成好几个 kernel（pow → mean → rsqrt → mul → mul），
  每个 kernel 都要把数据从显存读一遍写一遍 → 融合后只需 1 次读 + 1 次写。
  这就是 LLM 推理框架里"fused kernel"能带来 20%~50% 提速的原因。
""")

section("4) 归约（reduction）的三种写法对比")
print("""
  写法                          特点
  -----------------------------------------------------------------------------
  atomicAdd 到全局内存          最简单，但原子操作串行，元素多时很慢
  shared memory + 树形归约      经典写法，需要 __syncthreads()，注意 bank conflict
  warp shuffle（__shfl_*）     最快：数据在寄存器间传递，不占用 shared、无需同步
  cooperative groups (reduce)   C++ 封装，可读性最好，性能与上一种相当

  实战建议：
    · 先做 warp 级归约（__shfl_xor_sync 蝴蝶交换），再做跨 warp
    · shared memory 数组大小 = warp 数；注意避免 bank conflict（padding 或改变索引方式）
    · 归约一定要用 float/double 累加，避免半精度累加的误差爆炸
""")

section("5) 把这个 kernel 做得更快还能做什么")
print("""
  ① 一个 block 处理多行（减少 block 启动开销，d 较小时收益明显）
  ② 用 float4 向量化读写（d 能被 4 整除时）
  ③ 支持 fp16/bf16 输入，累加仍用 fp32（模板 + AT_DISPATCH）
  ④ 反向 kernel：保存 inv（或直接重算），dL/dx 需要两次归约（见论文/源码）
  ⑤ 与残差相加融合（residual + rmsnorm），省掉一次整块读写 —— 真实框架必做
  ⑥ 用 Persistent kernel：block 数固定为 SM × k，循环处理多行，避免尾部效应
""")
