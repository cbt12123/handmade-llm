# 第 7 章 · NVIDIA 算子编写

本目录的所有脚本在**没有 GPU 的机器上也能跑**（会打印 `[SKIP]` 并给出 CPU 版模拟），
在**有 NVIDIA 显卡 + nvcc** 的机器上会真正编译并执行 CUDA kernel。

## 0. 前置条件

| 组件 | 检查方式 | 说明 |
|---|---|---|
| NVIDIA 显卡 | `nvidia-smi` | 算力 ≥ 7.0 体验最好（有 TensorCore） |
| CUDA Toolkit | `nvcc --version` | 需要 **devel** 镜像（runtime 镜像没有 nvcc） |
| PyTorch(CUDA) | `python -c "import torch;print(torch.cuda.is_available())"` | 提供 extension 加载机制 |
| triton | `python -c "import triton"` | torch 2.x 通常自带 |

最省事的获得方式（就是仓库根目录那两个 Dockerfile）：

```bash
docker build -f docker/Dockerfile -t llm-line:cu121 .
docker run --rm -it --gpus all -v "$PWD":/workspace -w /workspace llm-line:cu121 bash
```

## 1. 文件清单

| 文件 | 内容 | 需要 GPU |
|---|---|---|
| `00_cuda_env_check.py` | 环境自检：GPU 型号/算力/显存/nvcc | 否 |
| `01_vector_add.cu` | 纯 CUDA 的 "Hello World"（含 CHECK 宏、Event 计时、block 大小对比） | 编译需要 nvcc |
| `01_vector_add_run.py` | 编译运行上面的 .cu；无 GPU 时用 Python 模拟线程索引 | 编译需要 nvcc |
| `02_torch_extension_gelu.py` | 用 `load_inline` 写 PyTorch CUDA 扩展（含 float4 向量化版本 + 基准） | 是 |
| `03_rms_norm_kernel.py` | 融合 RMSNorm：warp shuffle + shared memory 归约 | 是 |
| `04_softmax_kernel.py` | 行 softmax + **在线 softmax**（FlashAttention 的核心算法） | 是 |
| `05_matmul_tiled.cu` | GEMM 三连：朴素 → 分块 16×16 → 分块 32 + 2×2，并与 cuBLAS 对比 | 编译需要 nvcc |
| `05_matmul_tiled_run.py` | 编译运行；无 GPU 时用 numpy 定量统计访存次数 | 编译需要 nvcc |
| `06_triton_intro.py` | Triton 版 GELU / softmax / GEMM + autotune | 是 |
| `07_profiling_roofline.py` | 实测带宽与算力、Roofline 判定、nsys/ncu/torch.profiler 用法 | 部分是 |
| `08_cpu_simulation.py` | **纯 CPU**：模拟内存合并、bank 冲突、warp 分化、occupancy 估算 | 否 |

## 2. 手动编译命令备忘

```bash
# 查自己的算力（sm_xx）：A100=80, V100=70, RTX3090=86, RTX4090=89, H100=90
nvidia-smi --query-gpu=compute_cap --format=csv

cd script/07_nvidia_kernels
nvcc -O3 -arch=sm_80 01_vector_add.cu -o build/vector_add && ./build/vector_add
nvcc -O3 -arch=sm_80 05_matmul_tiled.cu -o build/matmul -lcublas && ./build/matmul

# 看寄存器/shared 占用（调优必看）
nvcc -O3 -arch=sm_80 -Xptxas -v 05_matmul_tiled.cu -o build/matmul -lcublas

# 性能分析
nsys profile -o report ./build/matmul
ncu --set full ./build/matmul
ncu --metrics sm__throughput.avg.pct_of_peak,dram__throughput.avg.pct_of_peak ./build/matmul

# 正确性/越界检查
compute-sanitizer ./build/vector_add
```

## 3. 常见坑

1. **算力不匹配**：报错 `no kernel image is available for execution`。
   解决：`export TORCH_CUDA_ARCH_LIST="8.6"`（或 nvcc 加 `-arch=sm_86`）。
2. **PyTorch extension 编译慢**：第一次会编译几分钟，产物缓存在 `~/.cache/torch_extensions`，
   容器里建议把这个目录挂成 volume（见 `docker/docker-compose.yml`）。
3. **`--use_fast_math`**：会牺牲精度（exp/sin 走近似指令）。写数值敏感的算子时不要加。
4. **忘记同步**：kernel 是异步的，计时必须 `torch.cuda.synchronize()` 或用 `cudaEvent`。
5. **dtype 不匹配**：numpy 默认 float64，torch 模型默认 float32，混用会报 dtype 错误。
6. **非连续 tensor**：`x.view()` 失败或结果错误时先 `x.contiguous()`，或在 kernel 里处理 stride。

## 4. 学习顺序建议

1. `00` → `08`（先建立概念，不需要 GPU）
2. `01` → `02`（第一个 kernel、接进 PyTorch）
3. `03` → `04`（归约与融合，LLM 里最常见的两类算子）
4. `05`（GEMM：理解"为什么优化 = 减少访存"）
5. `06`（Triton：真正干活时的首选工具）
6. `07`（性能分析：判断优化方向）
