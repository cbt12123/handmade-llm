"""
00 · 环境自检：你的机器能不能写 CUDA 算子？
============================================
跑这一节之前先跑它。需要三样东西都齐：
  ① NVIDIA 显卡（算力 ≥ 7.0 比较舒服；6.x 也行但没 TensorCore）
  ② CUDA Toolkit（提供 nvcc）—— 用 nvidia/cuda:xx-devel 镜像最省事
  ③ PyTorch（带 CUDA 支持）—— 提供 torch::Tensor 与 extension 加载机制

本脚本还会打印 GPU 的算力（sm_xx），后面编译 kernel 时要用到它。
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from _cuda_helpers import cuda_available, has_torch, nvcc_available, print_env
from common_utils import section, subsection

section("1) 环境信息")
print_env()

section("2) 三要素检查")
checks = [
    ("PyTorch", has_torch()),
    ("CUDA 可用（驱动 + GPU）", cuda_available()),
    ("nvcc（CUDA Toolkit）", nvcc_available()),
]
for name, ok in checks:
    print(f"  [{'OK' if ok else 'MISSING'}] {name}")
missing = [n for n, ok in checks if not ok]
if missing:
    print(f"\n  缺少: {', '.join(missing)}")
    print("""
  补齐方式：
    pip install torch                                  # PyTorch（本机 CPU 版也能跑通 1~6 章）
    docker build -f docker/Dockerfile -t llm-line:cu121 .   # 推荐：一次拿到 nvcc + torch + GPU
    docker run --rm -it --gpus all -v "$PWD":/workspace -w /workspace llm-line:cu121 bash

  没有 GPU 也能学：本章每个脚本都有 CPU 模拟版本，会打印 [SKIP] 并给出等价的 numpy 演示。
""")
else:
    print("\n  环境齐全，可以开始写算子了。")

section("3) 编译 kernel 时最重要的几个参数")
print("""
  -arch=sm_80        : 指定算力（A100=80, RTX3090=86, RTX4090=89, H100=90, V100=70）
                       不指定会按 nvcc 默认值编译，可能导致"运行时找不到镜像"
  TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9;9.0   : PyTorch extension 会为列表里的每个算力都编一份
  --use_fast_math    : 牺牲精度换速度（sin/exp 等用近似指令）
  -lineinfo          : 保留行号信息，nsys/ncu 才能把采样对应到源码行
  -Xptxas -v         : 打印寄存器用量与 shared memory 占用（调优必备）
""")

subsection("4) GPU 硬件概念速查（后面每个 kernel 都会用到）")
print("""
  层次结构：Grid → Block（CTA）→ Warp（32 线程）→ Thread
    · 一个 Block 最多 1024 线程，调度到【一个 SM】上执行
    · Warp 是真正的调度单位：32 个线程【锁步】执行同一条指令
    · 同一个 Block 内的线程可以共享 Shared Memory（片上，~100 倍快于显存）

  内存层次（速度从快到慢）：
    寄存器（每线程私有）> Shared Memory / L1（每 Block）> L2 > HBM（显存）
  典型带宽：HBM 约 1.5~3 TB/s，Shared Memory 约 10~20 TB/s（按 SM 聚合）
  → 算子优化的本质：尽量减少对 HBM 的访问次数，把数据留在片上复用
""")
