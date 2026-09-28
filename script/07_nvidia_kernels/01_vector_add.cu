// =============================================================================
// 01 · CUDA 版 "Hello World"：向量加法
// -----------------------------------------------------------------------------
// 这个文件是【纯 CUDA C++】，不依赖 PyTorch，可以直接用 nvcc 编译运行：
//
//   nvcc -O3 -arch=sm_80 01_vector_add.cu -o vector_add && ./vector_add
//
// 你会在这份代码里看到 CUDA 程序的骨架：
//   ① 在 host（CPU）上准备数据
//   ② cudaMalloc + cudaMemcpy 把数据搬到 device（GPU）
//   ③ 启动 kernel：kernel<<<grid, block>>>(...)
//   ④ cudaMemcpy 把结果搬回来 + cudaFree
//   ⑤ 每一步都用 CHECK 宏检查错误（CUDA 的异步特性让错误会"延迟爆发"）
// =============================================================================
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// 错误检查宏：CUDA 的调用都是异步的，不加检查的话错误会在很后面才暴露
// ---------------------------------------------------------------------------
#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d : %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Kernel：每个线程算一个元素
//   blockIdx.x  : 当前是第几个 block
//   blockDim.x  : 每个 block 有多少线程（这里是 256）
//   threadIdx.x : 当前线程在 block 内的编号
// 全局索引 i = blockIdx.x * blockDim.x + threadIdx.x
// ---------------------------------------------------------------------------
__global__ void vector_add(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c,
                           int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {                       // 边界检查：总线程数通常是 n 的上取整
        c[i] = a[i] + b[i];
    }
}

// ---------------------------------------------------------------------------
// main：注意 kernel 是【异步】的 —— 想拿耗时必须用 cudaEvent 或 cudaDeviceSynchronize
// ---------------------------------------------------------------------------
int main() {
    const int n = 1 << 20;                       // 1M 个元素
    const size_t bytes = n * sizeof(float);

    // ① host 内存
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    for (int i = 0; i < n; ++i) {
        h_a[i] = (float)(i % 100) * 0.01f;
        h_b[i] = (float)(i % 7) * 0.5f;
    }

    // ② device 内存 + 数据拷贝
    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc(&d_a, bytes));
    CHECK(cudaMalloc(&d_b, bytes));
    CHECK(cudaMalloc(&d_c, bytes));
    CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // ③ 启动配置：256 线程/block，block 数 = ceil(n / 256)
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    printf("配置: n=%d, threads/block=%d, blocks=%d\n", n, threads, blocks);

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));

    // ④ 拷回结果
    CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // ⑤ 校验
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (fabsf(h_c[i] - (h_a[i] + h_b[i])) > 1e-5f) { ok = false; break; }
    }
    printf("结果校验: %s\n", ok ? "PASS" : "FAIL");

    // 有效带宽 = (读 2 × 写 1) × 字节数 / 时间
    float gb = 3.0f * bytes / (ms * 1e-3f) / 1e9f;
    printf("kernel 耗时: %.3f ms, 有效带宽: %.1f GB/s\n", ms, gb);
    printf("（A100 的 HBM 带宽上限约 1555 GB/s，这个 kernel 是纯访存受限的）\n");

    // 顺便演示：block 大小对性能的影响
    printf("\n不同 block 大小下的耗时（同一个算法，只改 blockDim）：\n");
    for (int t : {32, 64, 128, 256, 512, 1024}) {
        int b = (n + t - 1) / t;
        CHECK(cudaEventRecord(start));
        vector_add<<<b, t>>>(d_a, d_b, d_c, n);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf("  threads=%4d  耗时=%.4f ms  带宽=%.1f GB/s\n",
               t, ms, 3.0f * bytes / (ms * 1e-3f) / 1e9f);
    }

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));
    free(h_a); free(h_b); free(h_c);
    printf("\n提示: blockDim 太小 → 并行度不够；太大 → 寄存器/spill 压力，且尾部浪费。256 通常是甜点。\n");
    return 0;
}
