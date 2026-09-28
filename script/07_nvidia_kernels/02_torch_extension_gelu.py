"""
02 · 写一个能被 PyTorch 调用的 CUDA 算子：GELU
================================================
真实工作中你很少从头写 main()，而是把 kernel 包成 PyTorch 的 extension：
    torch.utils.cpp_extension.load_inline(...)   ← 边写边编，调试期最方便
    生产环境用 setup.py + CUDAExtension          ← 可打包分发

一个 extension 需要三样东西：
  ① CUDA kernel（__global__ 函数）
  ② C++ wrapper（分配输出 tensor、算 grid/block、启动 kernel）
  ③ pybind11 绑定（load_inline 会根据 functions 参数自动生成）

本脚本还演示了算子开发的两个必备环节：**正确性对比** 与 **性能基准**。
"""
import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np

from _cuda_helpers import cuda_available, has_torch, skip
from common_utils import check_close, section, set_seed, subsection

set_seed(1)

if not has_torch():
    skip("需要 PyTorch：pip install torch")

import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------------------
# CUDA 源码：kernel + wrapper 写在一个字符串里（实际项目中放进 .cu 文件更好维护）
# ----------------------------------------------------------------------------------
CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// tanh 近似的 GELU： 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
__device__ __forceinline__ float gelu_tanh(float x) {
    const float k = 0.7978845608028654f;    // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

// ---- kernel：每个线程处理一个元素（elementwise 算子的标准写法）----
__global__ void gelu_kernel(const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // 网格跨步循环（grid-stride loop）：线程数不够时复用同一批线程继续处理
    // 这样 block 数量就不必等于 n/256，启动开销更小、也更健壮
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) {
        y[i] = gelu_tanh(x[i]);
    }
}

// ---- 多算一个元素（ILP）：让每个线程一次处理 4 个 float（float4）----
// 好处：① 访存自动向量化（128bit 事务） ② 指令级并行
__global__ void gelu_kernel_vec4(const float4* __restrict__ x, float4* __restrict__ y, int n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n4; i += stride) {
        float4 v = x[i];
        y[i] = make_float4(gelu_tanh(v.x), gelu_tanh(v.y), gelu_tanh(v.z), gelu_tanh(v.w));
    }
}

// ---- C++ wrapper ----
torch::Tensor gelu_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "only float32 is supported in this demo");
    auto y = torch::empty_like(x);
    int n = x.numel();

    const int threads = 256;
    // 只开固定数量的 block（比如 4×SM 数），靠 grid-stride loop 覆盖全部元素
    int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int blocks = min((n + threads - 1) / threads, sm_count * 4);
    gelu_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();     // 一定要检查启动错误（异步！）
    return y;
}

torch::Tensor gelu_cuda_vec4(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32, "need float32 CUDA tensor");
    TORCH_CHECK(x.numel() % 4 == 0, "numel must be divisible by 4");
    auto y = torch::empty_like(x);
    int n4 = x.numel() / 4;
    const int threads = 256;
    int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int blocks = min((n4 + threads - 1) / threads, sm_count * 4);
    gelu_kernel_vec4<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(y.data_ptr<float>()), n4);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
"""

CPP_SRC = r"""
#include <torch/extension.h>
torch::Tensor gelu_cuda(torch::Tensor x);
torch::Tensor gelu_cuda_vec4(torch::Tensor x);
"""

section("1) 编译 CUDA extension")
if not cuda_available():
    skip("当前环境没有可用的 CUDA GPU。\n"
         "        在 NVIDIA 机器上（或 docker/Dockerfile 容器里）重跑本脚本即可真正编译运行。\n"
         "        下面的 CPU 部分依然会执行，展示 PyTorch reference 与数值对比。")
    mod = None
else:
    from torch.utils.cpp_extension import load_inline

    try:
        mod = load_inline(
            name="gelu_cuda_ext",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["gelu_cuda", "gelu_cuda_vec4"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
        print("  编译成功！模块:", mod)
    except Exception as e:                                   # 编译失败时也要能继续学习
        print(f"  编译失败: {type(e).__name__}: {str(e)[:300]}")
        print("  常见原因：算力不匹配 → 设置环境变量 TORCH_CUDA_ARCH_LIST=8.6 后重试")
        mod = None

section("2) CPU 参考实现（GELU 的两种写法）")


def gelu_exact(x):
    """精确 GELU：0.5x(1 + erf(x/√2))"""
    from math import sqrt
    return 0.5 * x * (1.0 + torch.erf(x / sqrt(2.0)))


def gelu_tanh_cpu(x):
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))


x_cpu = torch.randn(1024)
print(f"  tanh 近似 vs 精确 GELU 的最大误差 = "
      f"{(gelu_tanh_cpu(x_cpu) - gelu_exact(x_cpu)).abs().max().item():.2e}")
print("  误差在 1e-3 量级 —— 这就是为什么框架默认用 tanh 近似：快且够准。")
print(f"  torch 内置实现 vs 精确 GELU 的最大误差 = "
      f"{(F.gelu(x_cpu, approximate='tanh') - gelu_exact(x_cpu)).abs().max().item():.2e}")

section("3) 正确性对比（自定义算子 vs torch）")
if mod is not None:
    x_gpu = torch.randn(1 << 20, device="cuda")
    y_custom = mod.gelu_cuda(x_gpu)
    y_torch = F.gelu(x_gpu, approximate="tanh")
    check_close(y_custom, y_torch, tol=1e-6, name="自定义 GELU vs torch GELU")

    y_vec4 = mod.gelu_cuda_vec4(x_gpu)
    check_close(y_vec4, y_torch, tol=1e-6, name="float4 向量化版本 vs torch")
else:
    print("[SKIP] 没有编译出 CUDA 模块，跳过 GPU 验证与基准")

section("4) 性能基准：为什么一定要 benchmark")
if mod is not None:
    def bench(fn, x, iters=100):
        for _ in range(10):
            fn(x)                                        # warmup（GPU 首次调用有编译/初始化开销）
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(x)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000

    n = 1 << 24                                          # 16M 元素 = 64MB
    x = torch.randn(n, device="cuda")
    bytes_moved = 2 * x.numel() * 4 / 1e6                # 读 + 写
    print(f"  数据规模: {x.numel():,} 元素（{bytes_moved:.0f} MB）")
    for name, fn in [("torch.nn.functional.gelu", lambda t: F.gelu(t, approximate="tanh")),
                     ("自定义 gelu_cuda", mod.gelu_cuda),
                     ("自定义 gelu_cuda_vec4(float4)", mod.gelu_cuda_vec4)]:
        ms = bench(fn, x)
        print(f"    {name:<32}{ms:8.3f} ms   带宽 {bytes_moved / ms / 1e3:7.1f} GB/s")
    print("""
  解读：GELU 是【访存受限】算子（每个元素只读 1 次写 1 次，计算极少）
    → 优化目标是"把带宽吃满"，而不是"算得更快"
    → float4 版本把 4 个 float 合成一次 128bit 访问，能显著降低指令与事务数
""")

section("5) 工程化清单（写自己的算子时照着做）")
print("""
  ① 先写 torch 参考实现，再用【随机输入】做数值对比（注意 fp16/bf16 的容差要放宽）
  ② 测多种输入形状：极小（1 个元素）、非对齐（不能整除 4/8）、极大（>2^31 要用 int64 索引）
  ③ 用 cudaEvent 或 torch.cuda.Event 计时，记得 warmup + synchronize
  ④ 检查错误：C10_CUDA_KERNEL_LAUNCH_CHECK()（同步）或 cudaGetLastError()（异步后）
  ⑤ 支持非连续 tensor：x.contiguous() 或在 kernel 里处理 stride
  ⑥ 支持不同 dtype：至少 fp32 / fp16 / bf16（用模板 + AT_DISPATCH_FLOATING_TYPES_AND2）
  ⑦ 反向也要写：PyTorch 里用 torch::autograd::Function 包一层
  ⑧ 打包分发：setup.py + CUDAExtension，或者干脆用 Triton（见 06 节）
""")
