"""
08 · 无 GPU 也能动手：CPU 模拟 GPU 的三个核心机制
====================================================
这一节不需要显卡，用 numpy/纯 Python 把三个"看不见但决定性能"的机制量化出来：

  ① 内存合并（coalescing）  ：warp 内 32 个线程访问连续 vs 跳跃地址的差距
  ② Shared Memory Bank 冲突：为什么 stride=32 的访问会慢 32 倍
  ③ Warp 分化（divergence）：if/else 让 warp 串行执行两个分支

再加一个：④ 占用率（occupancy）的估算方法
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE))
sys.path.append(str(HERE.parent))          # 让 common_utils 可 import

import numpy as np

from common_utils import section, subsection

WARP = 32
LINE_BYTES = 128                 # 一个 L1/L2 cache line（sector）的大小
FLOAT_BYTES = 4
FLOATS_PER_LINE = LINE_BYTES // FLOAT_BYTES      # 32 个 float


# ----------------------------------------------------------------------------------
def transactions(addrs):
    """统计一次 warp 访问会触发多少个 cache line 事务。

    硬件把地址按 128 字节对齐切分，落在同一条 line 的访问会被合并。
    """
    lines = {a // FLOATS_PER_LINE for a in addrs}
    return len(lines)


section("1) 内存合并：连续访问 vs 跳跃访问")
base = 1024
patterns = {
    "stride=1（连续，理想）": [base + i * 1 for i in range(WARP)],
    "stride=2": [base + i * 2 for i in range(WARP)],
    "stride=4": [base + i * 4 for i in range(WARP)],
    "stride=32（最坏）": [base + i * 32 for i in range(WARP)],
    "随机地址": list(np.random.default_rng(0).integers(0, 100000, WARP)),
}
print(f"  {'访问模式':<24}{'cache line 事务数':>20}{'相对理想':>12}")
ideal = None
for name, addrs in patterns.items():
    t = transactions(addrs)
    if ideal is None:
        ideal = t
    print(f"  {name:<24}{t:>20}{t / ideal:>12.1f}×")
print("""
  结论：跳跃访问会让带宽利用率成倍下降
  → 写 kernel 时第一原则：让 warp 内相邻线程访问相邻地址（threadIdx.x 对应最快变化维度）
  → 矩阵乘法里 B 矩阵按列访问就是"最坏情况"，所以必须转置或分块到 shared 再重排
""")

# ----------------------------------------------------------------------------------
section("2) Shared Memory Bank 冲突")
NUM_BANKS = 32


def bank_conflict_factor(stride, num_banks=NUM_BANKS):
    """32 个线程按 stride 访问 shared 数组，返回需要串行化的次数。"""
    banks = [(i * stride) % num_banks for i in range(WARP)]
    # 同一个 bank 被多个线程访问 → 必须串行；返回最大冲突次数
    counts = {}
    for b in banks:
        counts[b] = counts.get(b, 0) + 1
    return max(counts.values())


print(f"  {'访问步长':<16}{'最大 bank 冲突次数':>22}{'说明':>6}")
for stride in [1, 2, 3, 4, 8, 16, 32, 33]:
    f = bank_conflict_factor(stride)
    note = "" if f == 1 else "  ← 串行化"
    print(f"  stride={stride:<9}{f:>22}{note}")
print("""
  Shared memory 被切成 32 个 bank（每个 bank 4 字节，一个 warp 周期能服务 32 个线程）
    · stride=1：32 个线程落到 32 个不同 bank → 1 次搞定（最快）
    · stride=32：全部落到同一个 bank → 32 次串行（慢 32 倍！）
    · stride=33：又变回无冲突（经典 padding 技巧：数组开成 [TILE][TILE+1]）
  实战：矩阵转置、归约、写 tile 时最容易踩到，用 ncu 的 shared_ld_bank_conflict 指标查看
""")

# ----------------------------------------------------------------------------------
section("3) Warp 分化（divergence）")
print("""
  Warp 内的 32 个线程【锁步执行同一条指令】。如果遇到 if/else：
    · 所有线程都走同一分支 → 无代价
    · 一部分走 if、一部分走 else → 两个分支【串行执行】，另一半线程在等待
  代价 = (分支1 耗时 + 分支2 耗时)，而不是 max(两者)
""")


def simulate_divergence(condition_pattern, cost_if=10, cost_else=100):
    """condition_pattern: 32 个 bool，表示每个线程是否满足 if 条件"""
    take_if = sum(condition_pattern)
    take_else = len(condition_pattern) - take_if
    total = 0
    if take_if:
        total += cost_if                     # 整个 warp 执行一次 if 分支
    if take_else:
        total += cost_else                   # 再执行一次 else 分支
    return total, take_if, take_else


cases = {
    "全部走 if":            [True] * 32,
    "全部走 else":          [False] * 32,
    "前 16 个走 if":        [True] * 16 + [False] * 16,
    "只有 1 个走 if":       [True] + [False] * 31,
}
print(f"  {'分支分布':<20}{'if 线程数':>10}{'else 线程数':>12}{'总耗时(相对)':>14}")
for name, pat in cases.items():
    total, ti, te = simulate_divergence(pat)
    print(f"  {name:<20}{ti:>10}{te:>12}{total:>14}")
print("""
  对策：
    · 用 tl.where / 三元表达式代替分支（Triton 里是向量化的）
    · 把分支条件改成"对整个 warp 一致"（例如按 blockIdx 判断而不是 threadIdx）
    · 用掩码 + 算术代替分支：y = mask ? a : b  →  y = mask * a + (1 - mask) * b
""")

# ----------------------------------------------------------------------------------
section("4) 占用率（Occupancy）估算")
print("""
  Occupancy = 实际活跃 warp 数 / SM 支持的最大 warp 数
  三个限制因素（取最小值）：
    ① 每 SM 最大 warp 数（A100: 64，即 2048 线程）
    ② 每 SM 最大 block 数 × 每 block 的 warp 数（A100: 32 blocks/SM）
    ③ 寄存器：每 SM 64K 个 32-bit 寄存器；若每线程用 R 个，则
       每 block 线程数 T → 每 block 需要 R·T 个寄存器 → 每 SM 能放 65536/(R·T) 个 block
    ④ shared memory：每 SM 164KB（A100）；同理按每 block 的用量计算
""")


def occupancy(threads_per_block, regs_per_thread, smem_per_block_bytes,
              max_warps_per_sm=64, max_blocks_per_sm=32,
              regs_per_sm=65536, smem_per_sm=164 * 1024):
    warps_per_block = threads_per_block // 32
    limit_warp = max_warps_per_sm // warps_per_block
    limit_block = max_blocks_per_sm
    limit_reg = int(regs_per_sm // (regs_per_thread * threads_per_block)) if regs_per_thread else 99
    limit_smem = int(smem_per_sm // max(smem_per_block_bytes, 1))
    blocks = min(limit_warp, limit_block, limit_reg, limit_smem)
    active_warps = blocks * warps_per_block
    return blocks, active_warps, active_warps / max_warps_per_sm, dict(
        按warp上限=limit_warp, 按block上限=limit_block, 按寄存器=limit_reg, 按shared=limit_smem)


print(f"  {'配置':<40}{'blocks/SM':>10}{'活跃warps':>10}{'Occupancy':>11}")
configs = [
    ("256 线程, 32 寄存器, 0 shared", 256, 32, 0),
    ("256 线程, 64 寄存器, 0 shared", 256, 64, 0),
    ("256 线程, 128 寄存器, 0 shared", 256, 128, 0),
    ("128 线程, 40 寄存器, 8KB shared", 128, 40, 8 * 1024),
    ("256 线程, 40 寄存器, 32KB shared", 256, 40, 32 * 1024),
    ("512 线程, 40 寄存器, 16KB shared", 512, 40, 16 * 1024),
]
for name, t, r, s in configs:
    b, w, occ, lim = occupancy(t, r, s)
    print(f"  {name:<40}{b:>10}{w:>10}{occ:>11.0%}")
print("""
  目标：Occupancy ≥ 50%（能较好地隐藏访存延迟）；但 occupancy 高不等于性能好 ——
  如果每个线程干得太少（比如只做一次乘加），反而会被访存带宽卡死。
  权衡：寄存器用得多 → occupancy 低，但每个线程能算更多（ILP 更高）
  查看真实值：ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active
""")

section("5) 小结：把模拟结论搬到真机上")
print("""
  本节的四个机制在任何 NVIDIA 显卡上都成立（Ampere/Hopper/Blackwell 只是参数不同）。
  下一步（有 GPU 时）：
    1. python3 script/07_nvidia_kernels/00_cuda_env_check.py     确认算力与工具链
    2. python3 script/07_nvidia_kernels/01_vector_add_run.py     编译运行第一个 kernel
    3. python3 script/07_nvidia_kernels/02_torch_extension_gelu.py  写进 PyTorch
    4. python3 script/07_nvidia_kernels/05_matmul_tiled_run.py   GEMM 分块 + cuBLAS 对比
    5. ncu --set full ./build/matmul                             看真实指标
""")
