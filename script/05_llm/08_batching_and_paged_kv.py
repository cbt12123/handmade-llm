"""
08 · 推理服务的核心：连续批处理 + PagedAttention
==================================================
两个独立的优化，vLLM 把它们合在一起后把吞吐提升了数倍：

  ① 连续批处理（Continuous Batching / in-flight batching）
     静态批处理：等凑够一批 → 一起跑 → 整批都跑完才能放新请求（快的要等慢的）
     连续批处理：每个请求生成完一个 token 就让位，立刻插入新请求 → GPU 几乎不空转

  ② PagedAttention
     KV Cache 预分配"最大长度"会浪费大量显存（实际长度往往只有 1/3）
     借鉴操作系统虚拟内存：把 KV Cache 切成固定大小的 block，用页表按需分配
     → 显存利用率接近 100%，同样的卡能服务 2~4 倍并发

本脚本用纯 Python 模拟这两种机制（不依赖 GPU）。
"""
import sys
from pathlib import Path
import random

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common_utils import bytes_str, section, set_seed, subsection

set_seed(31)

# ----------------------------------------------------------------------------------
section("1) 静态批处理 vs 连续批处理（时间线模拟）")
REQUESTS = [                      # (到达时间, prompt 长度, 需要生成的 token 数)
    (0, 120, 40), (0, 80, 120), (0, 200, 20), (0, 60, 90), (0, 150, 60),
]
STEP = 1.0                        # 每个 decode step 的耗时（简化：与 batch 大小无关）


def simulate_static(requests, max_batch=4):
    """静态批处理：一次最多装 max_batch 个，整批跑完才放下一批。"""
    queue = sorted(requests, key=lambda r: r[0])
    time = 0.0
    done = []
    batch_no = 0
    while queue:
        batch = queue[:max_batch]
        queue = queue[max_batch:]
        batch_no += 1
        max_gen = max(r[2] for r in batch)
        # prefill 一次性算完
        time += max(r[1] for r in batch) * 0.01
        for r in batch:
            done.append((r, time + r[2] * STEP))
        time += max_gen * STEP
    return done, time


def simulate_continuous(requests, max_batch=4):
    """连续批处理：每个 step 检查谁完成了，立刻补进新请求。"""
    pending = sorted(requests, key=lambda r: r[0])
    running = []                   # [(req, 剩余生成数, 开始时间)]
    time = 0.0
    done = []
    while pending or running:
        while pending and len(running) < max_batch and pending[0][0] <= time:
            r = pending.pop(0)
            running.append([r, r[2], time])
        if not running:
            time = pending[0][0] if pending else time + STEP
            continue
        time += STEP
        still = []
        for req, left, start in running:
            left -= 1
            if left <= 0:
                done.append((req, time))
            else:
                still.append([req, left, start])
        running = still
    return done, time


done_s, t_s = simulate_static(REQUESTS)
done_c, t_c = simulate_continuous(REQUESTS)
print(f"  请求（prompt, 生成数）: {[(r[1], r[2]) for r in REQUESTS]}")
print(f"\n  静态批处理总耗时   = {t_s:.0f} 步")
print(f"  连续批处理总耗时   = {t_c:.0f} 步   吞吐提升 ≈ {t_s / t_c:.2f}×")
lat_s = [t for _, t in done_s]
lat_c = [t for _, t in done_c]
print(f"  静态：各请求完成时刻 = {[round(x) for x in lat_s]}")
print(f"  连续：各请求完成时刻 = {[round(x) for x in lat_c]}")
print("""
  观察：静态批处理里"生成 20 个 token"的请求要陪着"生成 120 个"的一起等 ——
  这就是所谓的 head-of-line blocking。连续批处理让短请求先走，平均延迟显著降低。
""")

# ----------------------------------------------------------------------------------
section("2) PagedAttention：显存碎片的账")
BLOCK = 16                                       # 每个 block 装 16 个 token 的 KV
MAX_LEN = 2048


def bytes_per_token(num_layers=32, num_kv_heads=8, head_dim=128, dtype_bytes=2):
    return num_layers * 2 * num_kv_heads * head_dim * dtype_bytes


BPT = bytes_per_token()
requests_len = [random.Random(0).randint(64, 900) for _ in range(24)]
print(f"  24 个请求，实际 prompt+输出长度：{sorted(requests_len)[:6]} ... {sorted(requests_len)[-3:]}")
print(f"  每 token 的 KV 显存 = {BPT} 字节（32 层 × 8 KV 头 × 128 维 × K/V × fp16）")

naive = len(requests_len) * MAX_LEN * BPT
used = sum(requests_len) * BPT
paged = sum((L + BLOCK - 1) // BLOCK * BLOCK for L in requests_len) * BPT
print(f"\n  预分配最大长度  : {bytes_str(naive)}   （利用率 {used / naive:.1%}）")
print(f"  实际需要        : {bytes_str(used)}")
print(f"  PagedAttention  : {bytes_str(paged)}   （利用率 {used / paged:.1%}）")
print(f"  节省显存        : {bytes_str(naive - paged)}  → 可多跑 {(naive / paged - 1) * 100:.0f}% 的并发")

section("3) 页表是怎么工作的")
class BlockTable:
    """简化版页表：逻辑块 → 物理块。"""

    def __init__(self, block_size, num_blocks):
        self.block_size = block_size
        self.free = list(range(num_blocks))
        self.table = {}                      # req_id -> [物理块号]

    def allocate(self, req_id, n_tokens):
        need = (n_tokens + self.block_size - 1) // self.block_size
        blocks = [self.free.pop(0) for _ in range(need)]
        self.table[req_id] = blocks
        return blocks

    def append(self, req_id, used_tokens):
        """已用 token 数增长，必要时再申请一块。"""
        need = (used_tokens + self.block_size - 1) // self.block_size
        while len(self.table[req_id]) < need:
            self.table[req_id].append(self.free.pop(0))

    def free_req(self, req_id):
        self.free.extend(self.table.pop(req_id))

    def lookup(self, req_id, token_idx):
        return self.table[req_id][token_idx // self.block_size], token_idx % self.block_size


bt = BlockTable(BLOCK, num_blocks=200)
bt.allocate("req-A", 5)
bt.append("req-A", 33)
bt.allocate("req-B", 2)
print(f"  req-A (33 tokens) 占用物理块: {bt.table['req-A']}")
print(f"  req-B (2 tokens)  占用物理块: {bt.table['req-B']}")
print(f"  req-A 第 20 个 token → 物理块 {bt.lookup('req-A', 20)}")
print("""
  关键点：同一个请求的 KV 在物理显存里【不连续】，
  注意力 kernel 需要支持"按页表 gather"—— 这就是 vLLM 要自己写 CUDA kernel 的原因（第 7 章）。
""")

section("4) 前缀共享（prefix caching）：多轮对话/相同 system prompt 的杀手锏")
shared_prefix = 500                            # 公共 system prompt 长度
n_req = 32
per_req_unique = 300
without = n_req * (shared_prefix + per_req_unique) * BPT
with_cache = (shared_prefix + n_req * per_req_unique) * BPT
print(f"  {n_req} 个请求共享 {shared_prefix} token 的前缀：")
print(f"    不共享: {bytes_str(without)}")
print(f"    共享后: {bytes_str(with_cache)}   节省 {(1 - with_cache / without):.1%}")
print("""
  实现要点：block 的内容做 hash（前缀 token 序列的哈希），相同 hash 的 block 引用计数 +1。
  这也是为什么"把不变的 system prompt 放前面"是一个真正有用的工程技巧。
""")

section("5) 一份生产级配置清单")
print("""
  框架选择：
    vLLM        : 吞吐优先，PagedAttention + 连续批处理，社区最活跃
    TGI         : HuggingFace 生态，功能全（量化、猜测解码）
    TensorRT-LLM: NVIDIA 官方，极致性能（需要编译 engine）
    llama.cpp   : CPU / 端侧 / Mac（GGUF 量化）
  必调参数：
    max_num_batched_tokens  : prefill 的最大 token 数（决定首 token 延迟）
    gpu_memory_utilization  : 留给 KV Cache 的显存比例（0.85~0.95）
    enable_prefix_caching   : 前缀共享，多轮对话场景强烈建议开
    quantization            : fp8（H100）/ AWQ / GPTQ
    speculative decoding    : 小模型起草 + 大模型验证，2~3× 加速（不改变输出分布）
""")
