# 第 7 章 · NVIDIA 算子编写：往下沉一层

> 配套代码：`script/07_nvidia_kernels/`（含 `README.md` 与两个 `.cu` 源文件）
> 本章全部是可动手的实验：**没有 GPU 也能学**（CPU 模拟版会跑起来），有 GPU 则真正编译运行。

---

## 7.0 先搞清楚你在优化什么

### GPU 的层次结构

```
Grid → Block（CTA）→ Warp（32 线程，锁步执行）→ Thread
  · 一个 Block 调度到一个 SM，最多 1024 线程
  · Warp 是真正的调度单位
  · 同一 Block 内共享 Shared Memory（片上，比显存快 ~10 倍）
```

### 内存层次

| 层级 | 位置 | 速度 |
|---|---|---|
| 寄存器 | 每线程私有 | 最快 |
| Shared Memory / L1 | 每 Block | ~10~20 TB/s（按 SM 聚合） |
| L2 | 全 GPU | 中等 |
| HBM（显存） | 全 GPU | 1.5~3 TB/s |

> **算子优化的本质**：尽量减少对 HBM 的访问次数，把数据留在片上复用。
> 这句话解释了第 5 章的 FlashAttention、融合算子、量化加速，也解释了本节的 GEMM 分块。

---

## 7.1 第一个 CUDA 程序

完整源码（`01_vector_add.cu`，可编译运行）：

```cuda
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(call)                                                       \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s:%d : %s\n", __FILE__,          \
                    __LINE__, cudaGetErrorString(err));                   \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

__global__ void vector_add(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;      // 全局一维索引
    if (i < n) {                                        // 边界检查：总线程数是 n 的上取整
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 1 << 20;
    const size_t bytes = n * sizeof(float);
    float *h_a = (float*)malloc(bytes), *h_b = (float*)malloc(bytes), *h_c = (float*)malloc(bytes);
    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc(&d_a, bytes));                      // ① device 内存
    CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;      // ② 启动配置
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);   // ③ 启动（异步！）
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));                   // 必须同步才能读耗时
    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));   // ④ 拷回
    printf("kernel 耗时: %.3f ms, 有效带宽: %.1f GB/s\n", ms, 3.0f*bytes/(ms*1e-3f)/1e9f);
}
```

编译与运行：

```bash
cd script/07_nvidia_kernels
nvcc -O3 -arch=sm_80 01_vector_add.cu -o build/vector_add && ./build/vector_add
# 或者让脚本替你做（会自动探测算力）：
python3 01_vector_add_run.py
```

脚本会打印不同 `blockDim` 的对比。输出形式如下
（**具体数字取决于你的 GPU**，请以 `./build/vector_add` 的实际输出为准）：

```
配置: n=1048576, threads/block=256, blocks=4096
结果校验: PASS
kernel 耗时: X.XXX ms, 有效带宽: XXX.X GB/s

threads=  32  耗时=X.XXXX ms  带宽=XX.X GB/s      ← 并行度不够
threads= 256  耗时=X.XXXX ms  带宽=XXX.X GB/s     ← 甜点
threads=1024  耗时=X.XXXX ms  带宽=XXX.X GB/s
```

判读方法：把实测带宽除以你的卡的**理论带宽**（`nvidia-smi` 或设备属性里能查），
低于 60% 说明没做向量化或 grid 太小。

文件：`script/07_nvidia_kernels/01_vector_add.cu` + `01_vector_add_run.py`

```cuda
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

骨架（每一步都要检查错误，因为 CUDA 是异步的）：

```
① host 准备数据 → ② cudaMalloc + cudaMemcpy(H2D) → ③ kernel<<<grid, block>>>
④ cudaMemcpy(D2H) → ⑤ cudaFree
```

```bash
nvcc -O3 -arch=sm_80 01_vector_add.cu -o build/vector_add && ./build/vector_add
```

脚本会打印不同 `blockDim`（32/64/128/256/512/1024）下的耗时与有效带宽。
结论：太小 → 并行度不够；太大 → 寄存器压力与尾部浪费；**256 通常是甜点**。

必须理解的三个概念：

1. **索引映射**：`i = blockIdx.x * blockDim.x + threadIdx.x`
2. **内存合并（coalescing）**：warp 内 32 个线程访问连续的 32 个 float → 合并成 2 次 128B 事务；
   跳跃访问会退化成 32 次 → **带宽掉 16 倍**（`08_cpu_simulation.py` 有量化模拟）
3. **占用率（occupancy）**：活跃 warp / SM 支持的最大 warp，最好 > 50%

---

## 7.2 写进 PyTorch：Extension

CUDA 侧（kernel + wrapper 写在同一个字符串里）：

```cuda
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float gelu_tanh(float x) {
    const float k = 0.7978845608028654f;                  // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

// grid-stride loop：线程数不够时复用同一批线程继续处理
__global__ void gelu_kernel(const float* __restrict__ x, float* __restrict__ y, int n) {
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        y[i] = gelu_tanh(x[i]);
    }
}

// float4 版本：一次处理 4 个 float，访存自动向量化成 128bit 事务
__global__ void gelu_kernel_vec4(const float4* __restrict__ x, float4* __restrict__ y, int n4) {
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        float4 v = x[i];
        y[i] = make_float4(gelu_tanh(v.x), gelu_tanh(v.y), gelu_tanh(v.z), gelu_tanh(v.w));
    }
}

torch::Tensor gelu_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "input must be a CUDA tensor");
    auto y = torch::empty_like(x);
    int n = x.numel();
    const int threads = 256;
    int sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int blocks = min((n + threads - 1) / threads, sm * 4);   // block 数不必等于 n/256
    gelu_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();                          // 一定要检查（异步！）
    return y;
}
```

Python 侧（三样东西：声明、源码、函数名）：

```python
from torch.utils.cpp_extension import load_inline

mod = load_inline(
    name="gelu_cuda_ext",
    cpp_sources="torch::Tensor gelu_cuda(torch::Tensor);",   # 声明（给 C++ 编译器看）
    cuda_sources=CUDA_SRC,                                   # 定义（给 nvcc 看）
    functions=["gelu_cuda"],                                 # 自动生成 pybind11 绑定
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)
```

正确性与性能（必做的两步）：

```python
# ① 数值对比
y_custom = mod.gelu_cuda(x_gpu)
y_torch  = F.gelu(x_gpu, approximate="tanh")
check_close(y_custom, y_torch, tol=1e-6)

# ② 基准：warmup + synchronize，否则测的是"第一次编译"的时间
def bench(fn, x, iters=100):
    for _ in range(10):
        fn(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000
```

脚本会打印类似下面的表格（数字随 GPU 变化）：

```
                                耗时(ms)      带宽(GB/s)
torch.nn.functional.gelu          X.XXX         ~1XXX
自定义 gelu_cuda                  X.XXX         ~1XXX
自定义 gelu_cuda_vec4(float4)     X.XXX         ~1XXX   ← 通常最快
```

判读：GELU 是**访存受限**算子，看带宽而不是 GFLOPS。
如果你的自定义版带宽明显低于 torch（< 70%），八成是没做向量化或 grid 配得太小。

脚本：`script/07_nvidia_kernels/02_torch_extension_gelu.py`

```python
from torch.utils.cpp_extension import load_inline
mod = load_inline(
    name="gelu_cuda_ext",
    cpp_sources="torch::Tensor gelu_cuda(torch::Tensor);",     # 声明
    cuda_sources=CUDA_SRC,                                      # kernel + wrapper
    functions=["gelu_cuda"],                                    # 自动生成 pybind11
    extra_cuda_cflags=["-O3", "--use_fast_math"],
)
```

kernel 里两个值得学的写法：

```cuda
// ① grid-stride loop：线程数不够时复用同一批线程继续处理，block 数不必等于 n/256
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)

// ② float4 向量化：一次处理 4 个 float，访存变成 128bit 事务
float4 v = x[i];
y[i] = make_float4(gelu(v.x), gelu(v.y), gelu(v.z), gelu(v.w));
```

**算子开发三步曲**（照做）：

1. 先写 torch 参考实现 → 随机输入做**数值对比**
2. 测多种形状：极小、不能整除 4/8、超大（>2³¹ 要 int64 索引）
3. `cudaEvent` 计时 + warmup + synchronize，对比带宽/GFLOPS

编译失败最常见的原因：**算力不匹配** → `export TORCH_CUDA_ARCH_LIST="8.6"`。

---

## 7.3 融合算子：RMSNorm

```cuda
// warp 级归约：蝴蝶交换，5 步把 32 个数加成一个，全程在寄存器里
__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

// block 级归约：先 warp 内归约 → 各 warp 写 shared → 前 32 个线程再归约一次
__device__ __forceinline__ float block_sum(float v, float* shared) {
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) shared[wid] = v;
    __syncthreads();                       // 必须同步：否则可能读到别的 warp 还没写的值
    v = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) v = warp_sum(v);
    return v;
}

// 融合 RMSNorm：一个 block 处理一行，全局访存从 3 次降到 1 次
__global__ void rms_norm_kernel(const float* __restrict__ x,
                                const float* __restrict__ w,
                                float* __restrict__ y, int d, float eps) {
    int row = blockIdx.x;
    const float* xr = x + (size_t)row * d;
    float* yr = y + (size_t)row * d;
    extern __shared__ float shared[];

    float sum = 0.0f;                                       // 第一遍：平方和
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    sum = block_sum(sum, shared);
    float inv = rsqrtf(sum / (float)d + eps);               // rsqrtf 是硬件指令

    for (int i = threadIdx.x; i < d; i += blockDim.x)       // 第二遍：写回（命中 L1/L2）
        yr[i] = xr[i] * inv * w[i];
}
```

wrapper 里的两个细节：

```cuda
int threads = 256;
while (threads > 32 && threads > d) threads >>= 1;         // 线程数不必超过维度
size_t shared_bytes = (threads / 32) * sizeof(float);      // 动态 shared memory
rms_norm_kernel<<<rows, threads, shared_bytes>>>(...);     // ← 第三个参数是 shared 大小
```

脚本：`script/07_nvidia_kernels/03_rms_norm_kernel.py`

朴素写法要 3~5 个 kernel（pow → mean → rsqrt → mul → mul），每个都要把数据读写一遍。
融合后：**一个 block 负责一行**，全局访存从 3 次降到 1 次。

两个核心原语：

```cuda
// warp 级归约：寄存器间"蝴蝶交换"，5 步合成一个和，不需要 shared、不需要同步
for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);

// 跨 warp：各 warp 写 shared → __syncthreads() → 前 32 个线程再归约一次
if (lane == 0) shared[wid] = v;
__syncthreads();
```

归约写法对比：

| 写法 | 特点 |
|---|---|
| `atomicAdd` 到全局 | 最简单，但串行，元素多时慢 |
| shared + 树形归约 | 经典，需 `__syncthreads()`，注意 bank conflict |
| **warp shuffle** | 最快：寄存器传递，不占 shared、不需同步 |
| cooperative groups | 可读性最好 |

进一步：一个 block 处理多行、float4、支持 bf16（累加仍用 fp32）、
**与残差相加融合**（真实框架必做）、persistent kernel。

---

## 7.4 Softmax 与在线 softmax

行 softmax（三趟扫描，一个 block 一行）：

```cuda
__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int d) {
    const float* xr = x + (size_t)blockIdx.x * d;
    float* yr = y + (size_t)blockIdx.x * d;
    extern __shared__ float smem[];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;

    float m = -FLT_MAX;                                     // ① 求最大值（防止 exp 溢出）
    for (int i = threadIdx.x; i < d; i += blockDim.x) m = fmaxf(m, xr[i]);
    m = warp_max(m);
    if (lane == 0) smem[wid] = m;
    __syncthreads();
    m = (threadIdx.x < blockDim.x / 32) ? smem[lane] : -FLT_MAX;
    if (wid == 0) m = warp_max(m);

    float s = 0.0f;                                         // ② exp 与求和
    for (int i = threadIdx.x; i < d; i += blockDim.x) s += __expf(xr[i] - m);
    s = warp_sum(s);
    if (lane == 0) smem[wid] = s;
    __syncthreads();
    s = (threadIdx.x < blockDim.x / 32) ? smem[lane] : 0.0f;
    if (wid == 0) s = warp_sum(s);

    for (int i = threadIdx.x; i < d; i += blockDim.x)       // ③ 写回
        yr[i] = __expf(xr[i] - m) / s;
}
```

在线 softmax（FlashAttention 的算法核心，Python 版更好读）：

```python
def online_softmax(x, block_len=4):
    m, l = -np.inf, 0.0                         # running max / running sum
    for b in range(0, len(x), block_len):
        blk = x[b:b + block_len]
        m_new = max(m, blk.max())
        corr = np.exp(m - m_new) if np.isfinite(m) else 0.0     # ← 第一次迭代要特殊处理
        l = l * corr + np.exp(blk - m_new).sum()
        m = m_new
    return np.exp(x - m) / l
```

```
一次性 softmax 与在线 softmax 的最大差 = 1.110e-16     ← 完全等价
```

脚本：`script/07_nvidia_kernels/04_softmax_kernel.py`

```
softmax(x_i) = exp(x_i − max) / Σ exp(x_j − max)      # 减 max 是为了防止 exp 溢出
```

在线 softmax（FlashAttention 的核心）：

```
m_new = max(m, block_max)
corr  = exp(m − m_new)
l     = l·corr + Σ exp(block − m_new)
```

脚本用 numpy 验证了它与一次性 softmax **完全等价**（误差 ~1e-16），
CUDA 版也与 `torch.softmax` 对齐。

调优点：

- `d ≤ 32` 时可以"一个 warp 处理一行"，完全不需要 `__syncthreads`
- **bank conflict**：32 个 bank，stride=32 的访问会串行 32 倍；
  规避方法是 padding（数组开成 `[TILE][TILE+1]`）或用 xor 索引
- `__expf` 是快速近似（约 2 ulp）；`--use_fast_math` 会全局开启
- 第一次迭代的修正因子 `exp(-inf)=0` 要特殊处理

---

## 7.5 GEMM：为什么优化 = 减少访存

V1 朴素（每个线程读 A 的一行 + B 的一列 —— B 是按列访问，**完全不合并**）：

```cuda
__global__ void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
            sum += A[row * K + k] * B[k * N + col];        // ← 列访问，stride = N
        C[row * N + col] = sum;
    }
}
```

V2 分块（tile 搬进 shared 后被 256 个线程复用）：

```cuda
#define TILE 16
__global__ void matmul_tiled(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;
        // 边界用 0 填充，比在循环里判断更快
        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();                                    // 等所有线程搬完
#pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();                                    // 等所有线程算完，才能覆盖 shared
    }
    if (row < M && col < N) C[row * N + col] = sum;
}
```

用 cuBLAS 做参照系（**列优先**的坑：用 `Cᵀ = Bᵀ·Aᵀ` 的技巧）：

```cuda
cublasHandle_t handle; cublasCreate(&handle);
const float alpha = 1.0f, beta = 0.0f;
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
            &alpha, d_B, N, d_A, K, &beta, d_C, N);        // 行优先 C = A·B
```

编译与运行：

```bash
nvcc -O3 -arch=sm_80 05_matmul_tiled.cu -o build/matmul -lcublas && ./build/matmul
```

输出形式如下（**数字随 GPU 不同**，典型量级：V2 比 V1 快 5~10 倍，cuBLAS 再快 3~10 倍）：

```
V1 朴素          :   XX.XXX ms    XX.XX GFLOPS
V2 分块 16x16    :    X.XXX ms   XXX.XX GFLOPS    ← 访存减少 16 倍的直接收益
V3 分块 32 + 2x2 :    X.XXX ms   XXX.XX GFLOPS
V4 cuBLAS        :    X.XXX ms  XXXX.XX GFLOPS    ← 生产级基线
结果校验（抽查 5 个元素）: PASS
```

**不管数字是多少，结论不变**：V2 的加速全部来自"全局访存少了 TILE 倍"，
而这可以用纯 numpy 验证（不需要 GPU）：

```python
naive_reads = 2 * M * N * K                                   # 每个输出元素读 K 个 A + K 个 B
tiled_reads = M * K * (N // TILE) + K * N * (M // TILE)
print(naive_reads / tiled_reads)                              # → 16.0
```

文件：`script/07_nvidia_kernels/05_matmul_tiled.cu` + `05_matmul_tiled_run.py`

| 版本 | 全局访存 | 说明 |
|---|---|---|
| V1 朴素 | `2·M·N·K` | 每个线程读 A 一行 + B 一列，**B 是按列访问 → 完全不合并** |
| V2 分块 16×16 | `2·M·N·K / 16` | tile 搬进 shared 后被 256 个线程复用 |
| V3 分块 32 + 每线程 2×2 | 更少 | 提高算术强度（寄存器分块） |
| V4 cuBLAS | — | 生产基线 |

```cuda
__shared__ float As[TILE][TILE], Bs[TILE][TILE];
As[ty][tx] = (row < M && aCol < K) ? A[row*K + aCol] : 0.0f;   // 边界补 0
__syncthreads();
for (int k = 0; k < TILE; ++k) sum += As[ty][k] * Bs[k][tx];
__syncthreads();
```

**Roofline 视角**（`07_profiling_roofline.py` 会算）：

```
算术强度 AI = FLOPs / Bytes
朴素 GEMM: AI ≈ 0.5  → 远低于 ridge point → 被带宽卡死
分块 GEMM: AI ≈ 8    → 接近算力上限
```

cuBLAS 还快得多的原因：**TensorCore（mma/WGMMA）**、**双缓冲（cp.async / TMA）**、
寄存器级分块 + 指令调度、split-K、swizzle 避免 bank conflict。

> 生产环境直接用 cuBLAS / CUTLASS / TensorRT-LLM，别自己写 GEMM。
> 但**必须懂原理**，否则读不懂 ncu 报告、写不出融合算子。
> 真正值得自己写的是：框架没有的融合算子、特殊形状、量化 GEMM。

---

## 7.6 Triton：写算子的首选工具

Hello World（elementwise GELU，10 行）：

```python
import triton
import triton.language as tl

@triton.jit
def gelu_kernel(X, Y, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements                                 # 用掩码代替 if 分支
    x = tl.load(X + offsets, mask=mask, other=0.0)
    y = 0.5 * x * (1.0 + tl.math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    tl.store(Y + offsets, y, mask=mask)

def gelu_triton(x):
    y = torch.empty_like(x)
    gelu_kernel[(triton.cdiv(x.numel(), 1024),)](x, y, x.numel(), BLOCK_SIZE=1024, num_warps=4)
    return y
```

分块 GEMM（核心是那一行 `tl.dot`）：

```python
@triton.jit
def matmul_kernel(A, B, C, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(A + offs_m[:, None] * K + (k + offs_k)[None, :], mask=..., other=0.0)
        b = tl.load(B + (k + offs_k)[:, None] * N + offs_n[None, :], mask=..., other=0.0)
        acc += tl.dot(a, b)                                     # ← 编译成 TensorCore 指令
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc, mask=...)

def matmul_triton(a, b, bm=64, bn=64, bk=32):
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float32)
    matmul_kernel[(triton.cdiv(a.shape[0], bm), triton.cdiv(b.shape[1], bn))](
        a, b, c, a.shape[0], b.shape[1], a.shape[1],
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=4, num_stages=3)
    return c
```

autotune（让编译器帮你搜 tile）：

```python
@triton.autotune(
    configs=[triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=3),
             triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
             triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64}, num_warps=8, num_stages=4)],
    key=["M", "N", "K"],                                        # 按形状缓存最优配置
)
@triton.jit
def matmul_autotune(A, B, C, M, N, K,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    ...                                     # 函数体与上面的 matmul_kernel 完全相同
```

脚本会打印两者的对比（数字随 GPU 变化，典型结论如下）：

```
1024×1024×1024 GEMM：
  Triton (BLOCK=64x64x32) :  X.XXX ms   ~1XXX GFLOPS     ← 通常达到 cuBLAS 的 60%~90%
  torch.matmul (cuBLAS)   :  X.XXX ms   ~2XXX GFLOPS
```

脚本：`script/07_nvidia_kernels/06_triton_intro.py`

| CUDA 概念 | Triton 对应 |
|---|---|
| `blockIdx` | `program_id(axis)` |
| `blockDim` | `num_warps=4`（编译器决定） |
| `threadIdx` | `tl.arange(0, BLOCK)`（向量化索引） |
| `__syncthreads()` | 不需要（编译器保证依赖） |
| shared memory | 不需要显式管理 |
| `if (i < n)` | `mask=` 参数传给 `tl.load/tl.store` |

```python
@triton.jit
def matmul_kernel(A, B, C, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(A + offs_m[:, None]*K + (k+offs_k)[None, :], mask=..., other=0.0)
        b = tl.load(B + (k+offs_k)[:, None]*N + offs_n[None, :], mask=..., other=0.0)
        acc += tl.dot(a, b)                 # ← 编译成 TensorCore 指令
```

调优旋钮：`num_warps`（4/8）、`num_stages`（2~5，软件流水线级数）、
`@triton.autotune` 自动搜 tile、调试用 `TRITON_INTERPRET=1`。

不调优的 Triton 通常能达到 cuBLAS 的 60%~90%，调优后可接近甚至超过（特定形状）。
**用 1/10 的代码量换 80%~100% 的性能** —— 这是今天写算子的默认选择。

---

## 7.7 性能分析：判断"慢在哪"

脚本：`script/07_nvidia_kernels/07_profiling_roofline.py`

```
可达性能 = min(峰值算力, 峰值带宽 × 算术强度)
ridge point = 峰值算力 / 峰值带宽 ≈ 12.5 FLOP/Byte（A100 fp32）
```

| 算子 | 算术强度 | 判定 |
|---|---|---|
| 向量加法 / GELU | ~0.17 | 访存受限 |
| RMSNorm / LayerNorm | ~0.67 | 访存受限 |
| GEMM (M=N=K=4096) | ~1366 | 算力受限 |
| GEMM (decode, batch=1) | ~1 | 访存受限 |

**这解释了量化为什么加速 decode 而不加速 prefill**。

工具：

```bash
nsys profile -o report ./build/matmul          # 时间线：看空转、同步、memcpy
ncu --set full ./build/matmul                  # kernel 详解
ncu --metrics sm__throughput.avg.pct_of_peak,dram__throughput.avg.pct_of_peak ./build/matmul
compute-sanitizer ./build/vector_add           # 越界/竞态检测
```

看什么：

- `Compute (SM) Throughput` vs `Memory Throughput` → 判断 compute/memory bound
- `Achieved Occupancy` → 并行度
- `Registers Per Thread` / `Local Memory` → 有没有寄存器溢出（spill）
- `Shared Memory Bank Conflicts` → shared 访问模式

调优清单：

1. 先判断 compute-bound 还是 memory-bound
2. memory-bound → 融合算子 / float4 / 减少中间结果落盘 / 用片上内存
3. compute-bound → TensorCore / 低精度 / 减少分支 / 提高 tile 与 ILP
4. occupancy 低 → 减少寄存器用量、减小 shared 占用
5. launch 开销大 → 合并 kernel 或 **CUDA Graph**
6. 端到端仍慢 → 查 CPU 侧（tokenizer、调度、数据拷贝）

---

## 7.8 无 GPU 也能动手：CPU 模拟

内存合并（数一数要多少次 cache line 事务）：

```python
WARP, FLOATS_PER_LINE = 32, 32          # 128B cache line = 32 个 float

def transactions(addrs):
    return len({a // FLOATS_PER_LINE for a in addrs})

print(transactions([base + i * 1  for i in range(WARP)]))    # → 1   连续，理想
print(transactions([base + i * 2  for i in range(WARP)]))    # → 2
print(transactions([base + i * 32 for i in range(WARP)]))    # → 32  最坏，慢 32 倍
```

bank 冲突（32 个 bank，落到同一个 bank 就要串行）：

```python
def bank_conflict_factor(stride, num_banks=32):
    banks = [(i * stride) % num_banks for i in range(32)]
    counts = {}
    for b in banks:
        counts[b] = counts.get(b, 0) + 1
    return max(counts.values())          # 同一 bank 被多少个线程访问

bank_conflict_factor(1)      # → 1    无冲突
bank_conflict_factor(32)     # → 32   完全串行
bank_conflict_factor(33)     # → 1    padding 技巧：数组开成 [TILE][TILE+1] 就无冲突了
```

warp 分化（两个分支**串行执行**，耗时相加）：

```python
def divergence_cost(pattern, cost_if=10, cost_else=100):
    total = 0
    if any(pattern):        total += cost_if      # 整个 warp 执行一次 if
    if not all(pattern):    total += cost_else    # 再执行一次 else
    return total

divergence_cost([True]*32)                    # → 10    无代价
divergence_cost([True]*16 + [False]*16)       # → 110   两个分支都跑
divergence_cost([True] + [False]*31)          # → 110   只有一个线程走 if，代价一样！
```

occupancy 估算（四个限制取最小）：

```python
def occupancy(threads_per_block, regs_per_thread, smem_bytes,
              max_warps=64, max_blocks=32, regs_per_sm=65536, smem_per_sm=164*1024):
    warps = threads_per_block // 32
    blocks = min(max_warps // warps,
                 max_blocks,
                 regs_per_sm // (regs_per_thread * threads_per_block),
                 smem_per_sm // max(smem_bytes, 1))
    return blocks * warps / max_warps

occupancy(256, 32, 0)          # → 1.00   理想
occupancy(256, 128, 0)         # → 0.25   寄存器吃太紧
occupancy(256, 40, 32*1024)    # → 0.25   shared 吃太紧
```

脚本的真实输出：

```
访问模式                    cache line 事务数     相对理想
  stride=1（连续，理想）                  1         1.0×
  stride=32（最坏）                     32        32.0×
```

脚本：`script/07_nvidia_kernels/08_cpu_simulation.py`（**纯 Python，永远能跑**）

量化了四个"看不见但决定性能"的机制：

| 机制 | 结论 |
|---|---|
| 内存合并 | stride=1 → 1 个 cache line 事务；stride=32 → 32 个（**慢 32 倍**） |
| Bank 冲突 | stride=1 无冲突；stride=32 串行 32 次；**stride=33 又变无冲突**（padding 技巧） |
| Warp 分化 | 只要有一个线程走不同分支，两个分支会**串行执行**（耗时相加） |
| Occupancy | 由 warp 上限 / block 上限 / 寄存器 / shared 四者取最小决定，目标 ≥ 50% |

---

## 7.9 速记卡

```
· GPU 优化的本质：减少对 HBM 的访问，把数据留在片上复用
· 索引三步：blockIdx → blockDim → threadIdx；grid-stride loop 更健壮
· 第一准则：让 warp 内相邻线程访问相邻地址（合并访存）
· 归约优先用 warp shuffle（__shfl_xor_sync），跨 warp 才用 shared
· 融合算子 = 少读写几次全局显存（RMSNorm 从 3 次降到 1 次）
· 分块把 GEMM 的算术强度从 0.5 提到 8+ → 从带宽受限变成算力受限
· 在线 softmax 是 FlashAttention 的核心，且与一次性结果精确等价
· 写算子优先 Triton；CUDA 用于极致性能；GEMM 直接用 cuBLAS/CUTLASS
· 优化前先测：nsys 看时间线，ncu 看 kernel 指标，Roofline 判断方向
· 常见坑：算力不匹配、忘记 synchronize、dtype 不匹配、非连续 tensor
```

---

## 7.10 真实生态：选对后端比手写 kernel 更常见

```bash
python3 script/07_nvidia_kernels/09_real_backends.py    # pip install transformers
```

真实项目里，你更常做的是"选对注意力后端"，而不是自己写。

```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    dtype=torch.bfloat16,                      # A100/H100 用 bf16
    device_map="auto",                         # 多卡自动切分
    attn_implementation="flash_attention_2",   # eager / sdpa / flash_attention_2
)
model.config.use_cache = True
```

后端怎么选（脚本会检测你的环境）：

```
torch 版本          : 2.x
CUDA 可用           : False
flash-attn 已安装    : False
flash-attn 可用     : 否（需要 NVIDIA GPU，算力 ≥ 8.0）

eager :    2.402 ms/次
sdpa  :    1.904 ms/次    加速比 1.26×
```

规则：

```
· 没 GPU 或没装 flash-attn         → sdpa
· 有 GPU + flash-attn + bf16/fp16  → flash_attention_2（最快，省 O(T²) 显存）
· 需要 output_attentions=True      → 只能用 eager（做可视化时）
· 训练求稳                          → sdpa（flash 对变长/padding 更挑）
```

一行 torch.compile：

```python
compiled = torch.compile(model)                          # 默认
compiled = torch.compile(model, mode="reduce-overhead")  # 用 CUDA Graph 降启动开销
compiled = torch.compile(model, mode="max-autotune")     # 启动时自动调优
```

安装 flash-attn：

```bash
pip install flash-attn --no-build-isolation
```

加速手段的收益排序（LLM 推理）：

```
① 用对框架：vLLM / TGI（连续批处理 + PagedAttention）   3~10×
② KV Cache                                            5~20×（越长越明显）
③ 量化（W4A16 / fp8）                                  1.5~3×（decode 阶段）
④ FlashAttention                                       1.5~2× + 省 O(T²) 显存
⑤ torch.compile / CUDA Graph                           1.1~1.5×（小 batch 更明显）
⑥ 手写融合算子（RMSNorm+Residual 等）                   5%~20%
```

> 结论：先把①②③做对，再考虑写算子 —— 算子是"最后一公里"的优化。

---

## 7.11 代码索引

| 文件 | 内容 | 需要 GPU |
|---|---|---|
| `00_cuda_env_check.py` | GPU/算力/nvcc/torch 环境自检 | 否 |
| `01_vector_add.cu` | 纯 CUDA Hello World（CHECK 宏、Event 计时、block 对比） | 编译需 nvcc |
| `01_vector_add_run.py` | 编译运行 + CPU 版线程索引模拟 | 编译需 nvcc |
| `02_torch_extension_gelu.py` | `load_inline` 扩展（含 float4 版 + 基准） | 是 |
| `03_rms_norm_kernel.py` | 融合 RMSNorm：warp shuffle + shared 归约 | 是 |
| `04_softmax_kernel.py` | 行 softmax + 在线 softmax | 是 |
| `05_matmul_tiled.cu` | GEMM 三连 + cuBLAS 对比 | 编译需 nvcc |
| `05_matmul_tiled_run.py` | 编译运行 + numpy 访存次数统计 | 编译需 nvcc |
| `06_triton_intro.py` | Triton GELU / softmax / GEMM + autotune | 是 |
| `07_profiling_roofline.py` | 实测带宽算力、Roofline、nsys/ncu 用法 | 部分 |
| `08_cpu_simulation.py` | 纯 CPU 模拟四个核心机制 | 否 |

详细编译命令与排错见 `script/07_nvidia_kernels/README.md`。
