"""
01 · 编译并运行上面的 CUDA 向量加法（没有 GPU 时给出 CPU 模拟）
================================================================
有 nvcc + GPU：真正编译 01_vector_add.cu 并执行，打印带宽与 block 大小对比
没有 nvcc     ：用 numpy 演示同样的"索引计算 + 边界检查"逻辑，并解释每个 CUDA 概念
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np

from _cuda_helpers import compile_cu, ensure_dir, nvcc_available, run_binary, skip
from common_utils import bytes_str, section, subsection

HERE = Path(__file__).resolve().parent

section("1) 编译并运行 CUDA 程序")
if not nvcc_available():
    skip("未找到 nvcc。请在 docker/Dockerfile 构建的容器里运行：\n"
         "        docker run --rm -it --gpus all -v \"$PWD\":/workspace -w /workspace llm-line:cu121 \\\n"
         "               python3 script/07_nvidia_kernels/01_vector_add_run.py")
else:
    build = ensure_dir()
    exe = compile_cu(HERE / "01_vector_add.cu", build / "vector_add")
    print(run_binary(exe))

section("2) CPU 版模拟：把 CUDA 的线程模型用 Python 循环写出来")
N, THREADS = 1 << 12, 256


def cpu_vector_add(a, b, threads=THREADS):
    """模拟 <<<blocks, threads>>> 的启动方式：算一遍 global index，越界就跳过。"""
    n = a.size
    blocks = (n + threads - 1) // threads
    c = np.zeros_like(a)
    launch_stats = {"blocks": blocks, "threads": threads, "total_threads": blocks * threads,
                    "idle_threads": blocks * threads - n}
    for blockIdx in range(blocks):
        for threadIdx in range(threads):
            i = blockIdx * threads + threadIdx          # ← 就是 kernel 里那一行
            if i < n:                                    # ← 边界检查
                c[i] = a[i] + b[i]
    return c, launch_stats


a = np.arange(N, dtype=np.float32) * 0.01
b = np.arange(N, dtype=np.float32) * 0.5
c, stats = cpu_vector_add(a, np.full(N, 0.5, dtype=np.float32))
print(f"  启动配置: {stats}")
print(f"  前 5 个结果: {c[:5]}   （a[:5]={a[:5]}）")
print(f"  浪费的线程数 = {stats['idle_threads']} / {stats['total_threads']} "
      f"（{stats['idle_threads'] / stats['total_threads']:.1%}，n 不是 threads 的整数倍时必然发生）")

subsection("3) 三个必须理解的概念")
print("""
  ① 索引映射  i = blockIdx.x * blockDim.x + threadIdx.x
     二维数组还有 y/z 维度：i = (gridDim.x * blockIdx.y + ...) 依此类推
     矩阵乘法 kernel 会用到二维索引：row = blockIdx.y * blockDim.y + threadIdx.y

  ② 内存合并（coalescing）
     warp 内的 32 个线程如果访问【连续的 32 个 float】，硬件会合并成 2 次 128B 事务
     如果每个线程跳着访问（stride 大），会退化成 32 次单独事务 → 带宽掉 16 倍
     → 所以 kernel 里"让相邻线程访问相邻地址"是第一准则

  ③ 占用率（occupancy）
     occupancy = 活跃的 warp 数 / SM 支持的最大 warp 数
     受限于：每 Block 的线程数、寄存器用量、shared memory 用量
     经验：想要"隐藏访存延迟"，occupancy 最好 > 50%
     查看方式：ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active
""")

section("4) 常见错误与排查")
print("""
  · kernel 静默不执行 → 大概率是启动配置越界（grid 太大）或没检查 cudaGetLastError
  · 结果全 0          → 忘了 cudaMemcpy(D2H)，或在 kernel 里写了局部变量没写回指针
  · 非法内存访问      → 边界检查缺失 / 指针是 host 地址
  · 速度奇慢          → ① 没做内存合并 ② 每个线程干太少活 ③ 用 printf 调试忘删
  · 调试神器：
      cuda-gdb ./a.out                     # 源码级调试
      compute-sanitizer ./a.out            # 检测越界与竞态（替代旧的 cuda-memcheck）
      nsys profile ./a.out                 # 时间线
      ncu --set full ./a.out               # kernel 级详细指标
""")
