// =============================================================================
// 05 · 矩阵乘法（GEMM）：从朴素到分块，理解 GPU 算子优化的"第一性原理"
// -----------------------------------------------------------------------------
// 编译（需要 CUDA Toolkit；cuBLAS 对比需要 -lcublas）：
//   nvcc -O3 -arch=sm_80 05_matmul_tiled.cu -o matmul -lcublas && ./matmul
//
// 演进路线（每一步都在解决上一步的瓶颈）：
//   V1 朴素：每个线程读 A 的一行 + B 的一列 → 全局访存 O(M·N·K)，带宽被打爆
//   V2 分块：把 A/B 的小块搬进 shared memory 复用 → 全局访存降到 1/TILE
//   V3 更大的块 + 每线程算 2×2：提高计算/访存比（算术强度）
//   V4 cuBLAS / TensorCore：生产环境直接用（本文件用 cuBLAS 做参照系）
// =============================================================================
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK(call)                                                            \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d : %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CHECK_CUBLAS(call)                                                     \
    do {                                                                       \
        cublasStatus_t st = call;                                              \
        if (st != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", st, __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// V1 朴素实现：一个线程算 C 的一个元素
// 每个线程要读 A 的 K 个元素 + B 的 K 个元素 → 全局访存 2·M·N·K
// ---------------------------------------------------------------------------
__global__ void matmul_naive(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];   // B 是按列访问 → 不合并！
        }
        C[row * N + col] = sum;
    }
}

// ---------------------------------------------------------------------------
// V2 分块（tiling）：16×16 的 tile 搬进 shared memory，块内复用
// 全局访存: A 被读 N/TILE 次，B 被读 M/TILE 次 → 总量降到 1/TILE
// ---------------------------------------------------------------------------
#define TILE 16
__global__ void matmul_tiled(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;
        // 边界用 0 填充（比在循环里判断更快）
        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();                       // 等所有线程把 tile 搬完

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();                       // 等所有线程算完，才能覆盖 shared
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

// ---------------------------------------------------------------------------
// V3 32×32 tile + 每个线程算 2×2（寄存器分块，提高算术强度）
// ---------------------------------------------------------------------------
#define TILE2 32
__global__ void matmul_tiled_2x2(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[TILE2][TILE2];
    __shared__ float Bs[TILE2][TILE2];

    int baseRow = blockIdx.y * TILE2;
    int baseCol = blockIdx.x * TILE2;
    float acc[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

    for (int t = 0; t < (K + TILE2 - 1) / TILE2; ++t) {
        for (int m = 0; m < 2; ++m) {
            int r = baseRow + threadIdx.y * 2 + m;
            int c = t * TILE2 + threadIdx.x;
            As[threadIdx.y * 2 + m][threadIdx.x] = (r < M && c < K) ? A[r * K + c] : 0.0f;
        }
        for (int m = 0; m < 2; ++m) {
            int r = t * TILE2 + threadIdx.y;
            int c = baseCol + threadIdx.x * 2 + m;
            Bs[threadIdx.y][threadIdx.x * 2 + m] = (r < K && c < N) ? B[r * N + c] : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE2; ++k) {
            float a0 = As[threadIdx.y * 2][k], a1 = As[threadIdx.y * 2 + 1][k];
            float b0 = Bs[k][threadIdx.x * 2], b1 = Bs[k][threadIdx.x * 2 + 1];
            acc[0][0] += a0 * b0; acc[0][1] += a0 * b1;
            acc[1][0] += a1 * b0; acc[1][1] += a1 * b1;
        }
        __syncthreads();
    }
    for (int m = 0; m < 2; ++m) {
        for (int n = 0; n < 2; ++n) {
            int r = baseRow + threadIdx.y * 2 + m;
            int c = baseCol + threadIdx.x * 2 + n;
            if (r < M && c < N) C[r * N + c] = acc[m][n];
        }
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    // 为了让对比明显，用一个较大的方阵；小矩阵跑不出带宽瓶颈
    const int M = 1024, N = 1024, K = 1024;
    const size_t szA = (size_t)M * K * sizeof(float);
    const size_t szB = (size_t)K * N * sizeof(float);
    const size_t szC = (size_t)M * N * sizeof(float);

    float *h_A = (float*)malloc(szA), *h_B = (float*)malloc(szB), *h_C = (float*)malloc(szC);
    for (int i = 0; i < M * K; ++i) h_A[i] = (float)(i % 13) * 0.01f;
    for (int i = 0; i < K * N; ++i) h_B[i] = (float)(i % 17) * 0.01f;

    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc(&d_A, szA));
    CHECK(cudaMalloc(&d_B, szB));
    CHECK(cudaMalloc(&d_C, szC));
    CHECK(cudaMemcpy(d_A, h_A, szA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, szB, cudaMemcpyHostToDevice));

    cudaEvent_t s, e;
    CHECK(cudaEventCreate(&s));
    CHECK(cudaEventCreate(&e));
    float ms = 0.0f;
    double flops = 2.0 * M * N * K;

    // ---------------- V1 ----------------
    dim3 block1(16, 16), grid1((N + 15) / 16, (M + 15) / 16);
    CHECK(cudaEventRecord(s));
    matmul_naive<<<grid1, block1>>>(d_A, d_B, d_C, M, N, K);
    CHECK(cudaEventRecord(e));
    CHECK(cudaEventSynchronize(e));
    CHECK(cudaEventElapsedTime(&ms, s, e));
    printf("V1 朴素          : %8.3f ms  %8.2f GFLOPS\n", ms, flops / (ms * 1e-3) / 1e9);

    // ---------------- V2 ----------------
    dim3 block2(TILE, TILE), grid2((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    CHECK(cudaEventRecord(s));
    matmul_tiled<<<grid2, block2>>>(d_A, d_B, d_C, M, N, K);
    CHECK(cudaEventRecord(e));
    CHECK(cudaEventSynchronize(e));
    CHECK(cudaEventElapsedTime(&ms, s, e));
    printf("V2 分块 16x16    : %8.3f ms  %8.2f GFLOPS\n", ms, flops / (ms * 1e-3) / 1e9);

    // ---------------- V3 ----------------
    dim3 block3(TILE2 / 2, TILE2 / 2), grid3((N + TILE2 - 1) / TILE2, (M + TILE2 - 1) / TILE2);
    CHECK(cudaEventRecord(s));
    matmul_tiled_2x2<<<grid3, block3>>>(d_A, d_B, d_C, M, N, K);
    CHECK(cudaEventRecord(e));
    CHECK(cudaEventSynchronize(e));
    CHECK(cudaEventElapsedTime(&ms, s, e));
    printf("V3 分块 32 + 2x2 : %8.3f ms  %8.2f GFLOPS\n", ms, flops / (ms * 1e-3) / 1e9);

    // ---------------- V4 cuBLAS（参照系）----------------
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    const float alpha = 1.0f, beta = 0.0f;
    // 注意：cuBLAS 是列优先，利用 C^T = B^T · A^T 的技巧直接算行优先结果
    CHECK(cudaEventRecord(s));
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                             &alpha, d_B, N, d_A, K, &beta, d_C, N));
    CHECK(cudaEventRecord(e));
    CHECK(cudaEventSynchronize(e));
    CHECK(cudaEventElapsedTime(&ms, s, e));
    printf("V4 cuBLAS        : %8.3f ms  %8.2f GFLOPS   ← 生产级基线\n",
           ms, flops / (ms * 1e-3) / 1e9);

    // ---------------- 校验：与 CPU 结果对比 ----------------
    CHECK(cudaMemcpy(h_C, d_C, szC, cudaMemcpyDeviceToHost));
    bool ok = true;
    for (int i = 0; i < 5 && ok; ++i) {
        float expect = 0.0f;
        for (int k = 0; k < K; ++k) expect += h_A[i * K + k] * h_B[k * N + i];
        if (fabsf(expect - h_C[i * N + i]) > 1e-2f) ok = false;
    }
    printf("结果校验（抽查 5 个元素）: %s\n", ok ? "PASS" : "FAIL");

    printf("\n思考：本文件的 V2/V3 仍远慢于 cuBLAS，差在：\n"
           "  · 没有用 TensorCore（mma 指令 / WMMA / WGMMA）\n"
           "  · 没有做双缓冲（异步拷贝 global→shared，用 cp.async / TMA 隐藏延迟）\n"
           "  · 没有寄存器级分块 + 指令调度、没有 split-K、没有 swizzle 避免 bank conflict\n"
           "真实生产环境：直接用 cuBLAS / cuBLASLt / CUTLASS / TensorRT-LLM，别自己写 GEMM。\n");

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK(cudaEventDestroy(s));
    CHECK(cudaEventDestroy(e));
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C);
    return 0;
}
