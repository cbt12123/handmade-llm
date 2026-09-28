# 第 5 章 · LLM 核心技术：从"能跑"到"跑得起"

> 配套代码：`script/05_llm/`
> 本章回答三个工程问题：**要多少显存？怎么生成得又好又快？怎么低成本微调？**

---

## 5.1 先算账：Token 预算与显存

把估算写成函数（改参数就能算你自己的模型）：

```python
def kv_cache_bytes(num_layers, num_kv_heads, head_dim, seq_len, batch=1, dtype_bytes=2):
    """KV Cache：每层每头每个位置都要存 K 和 V 两个向量。"""
    return num_layers * 2 * batch * seq_len * num_kv_heads * head_dim * dtype_bytes

def model_weight_bytes(num_params, dtype_bytes=2):
    return num_params * dtype_bytes

def training_bytes(num_params, optimizer="adamw"):
    """混合精度：权重 fp16 + 梯度 fp16 + fp32 主权重 + Adam m/v = 16 字节/参数"""
    return num_params * (2 + 2 + 4 + 4 + 4)

def training_flops(num_params, num_tokens):
    return 6 * num_params * num_tokens          # 前向 2N + 反向 4N
```

脚本输出的真实显存表（fp16）：

```
模型              权重(fp16)   KV/token     4k 上下文    32k 上下文
Llama-3-8B          14.90GB     128.00KB     512.00MB      4.00GB
Llama-3-70B        130.39GB     320.00KB       1.25GB     10.00GB
Qwen2-7B            13.04GB      56.00KB     224.00MB      1.75GB
Mixtral-8x7B        86.99GB     128.00KB     512.00MB      4.00GB
```

训练算力（脚本输出，假设 A100 利用率 40%）：

```
模型            训练 token            FLOPs            A100·天
1B               20.00B         120000.00T             11.1
7B                 1.00T       42000000.00T           3895.1
```

> 数量级感受：7B 模型训 1T token ≈ 单卡跑十年，所以一定要用几十上百张卡并行。
> 这也是为什么"复现一个大模型"从来不是单机能做的事。

脚本：`script/05_llm/01_token_and_memory_budget.py`

### Token 预算（经验系数）

```
英文：1 token ≈ 4 字符 ≈ 0.75 单词
中文：1 token ≈ 1~1.5 汉字
代码：1 token ≈ 3~4 字符（缩进也算）
```

容易踩的坑：

- **上下文 = 输入 + 已生成的输出**，总长受窗口限制
- 多轮对话会累积历史：8k 窗口可能 10 轮就满
- **system prompt 每轮都要重算**，长 system prompt 的成本是"每轮"的

### 推理显存 = 权重 + KV Cache + 少量激活

```
KV Cache = 层数 × 2 × batch × seq_len × kv_heads × head_dim × dtype_bytes
```

实测（脚本输出）：

```
Llama-3-8B    权重 14.90GB   KV/token 128KB   4k → 512MB    32k → 4.00GB
Llama-3-70B   权重 130.39GB  KV/token 320KB   4k → 1.25GB   32k → 10.00GB
```

> 8B 模型、32k 上下文、batch=8 时，KV Cache 能轻松超过权重本身。

### 训练显存（混合精度 + AdamW）

```
每参数 ≈ 16 字节 = 权重 fp16(2) + 梯度 fp16(2) + fp32 主权重(4) + Adam m(4) + Adam v(4)
```

| 模型 | AdamW | 8-bit 优化器 |
|---|---|---|
| 7B | ~112 GB | ~70 GB |
| 70B | ~1.1 TB | ~700 GB |

省显存三板斧：**ZeRO-1/2/3**、**激活重计算**（+30% 算力换 5~10× 激活显存）、**8-bit 优化器**。

### 训练算力

```
FLOPs ≈ 6·N·D（前向 2N + 反向 4N）
Chinchilla 经验：最优训练 token ≈ 20 × 参数量
```

---

## 5.2 解码策略

全部手写（30 行搞定所有策略）：

```python
import numpy as np
rng = np.random.default_rng(0)

def probs(logits, temperature=1.0):
    z = logits / max(temperature, 1e-6)          # 温度：给 logits 做缩放
    e = np.exp(z - z.max())
    return e / e.sum()

def sample_greedy(z):
    return int(np.argmax(z))

def sample_top_k(z, k=5, temperature=1.0):
    p = probs(z, temperature)
    idx = np.argsort(-p)[:k]                     # 固定保留前 k 个
    return int(rng.choice(idx, p=p[idx] / p[idx].sum()))

def sample_top_p(z, p_thresh=0.9, temperature=1.0):
    """nucleus：按概率从高到低累加，直到达到 p —— 候选集大小是动态的。"""
    p = probs(z, temperature)
    order = np.argsort(-p)
    cutoff = int(np.searchsorted(np.cumsum(p[order]), p_thresh) + 1)
    idx = order[:cutoff]
    return int(rng.choice(idx, p=p[idx] / p[idx].sum()))

def sample_with_penalty(z, history, presence=0.5, frequency=0.5, temperature=1.0):
    z2 = z.copy()
    for w, cnt in history.items():               # 对已出现过的 token 直接从 logits 里减
        z2[w2i[w]] -= presence + frequency * cnt
    return int(rng.choice(V, p=probs(z2, temperature)))
```

同一个上下文 `'the'` 下不同温度的真实分布（脚本输出）：

```
T=0.3  熵=1.65  top5: model(0.549), units(0.150), basic(0.110), </s>(0.050), patterns(0.023)
T=0.7  熵=2.83  top5: model(0.193), units(0.111), basic(0.097), </s>(0.069), patterns(0.049)
T=1.0  熵=2.99  top5: model(0.132), units(0.089), basic(0.081), </s>(0.064), patterns(0.051)
T=2.0  熵=3.10  top5: model(0.078), units(0.064), basic(0.061), </s>(0.055), patterns(0.049)
```

> 温度的本质：T→0 变成贪心（分布尖锐），T→∞ 变成均匀分布（熵最大）。

top-p 的自适应性（脚本输出）：

```
top-k=5   固定保留前 5 个， 累积概率 = 0.376        ← 不管分布形状，永远是 5 个
top-p=0.5  动态保留前 8 个， 累积概率 = 0.550
top-p=0.8  动态保留前 15 个，累积概率 = 0.804
top-p=0.95 动态保留前 21 个，累积概率 = 0.963
```

脚本：`script/05_llm/02_sampling_strategies.py`

模型给出的是 logits → softmax → 下一个 token 的分布。"怎么选一个"决定质量与多样性。

| 策略 | 做法 | 适用 |
|---|---|---|
| Greedy | 每次取最大 | 确定性任务 |
| Temperature | `logits / T` | T<1 尖锐，T>1 平滑 |
| top-k | 只在概率最高的 k 个里采样 | 简单但 k 固定 |
| **top-p (nucleus)** | 累积概率达 p 的最小集合 | **自适应，最常用** |
| 重复惩罚 | 对已出现 token 减 `presence + freq·cnt` | 抑制复读 |
| beam search | 保留 m 条候选，最大化整句概率 | 翻译/摘要 |

**top-p 比 top-k 好的原因**：分布尖锐时只取 1~2 个候选，平坦时自动扩大候选集 —— 自适应。

推荐配置：

```
通用对话    T=0.7, top_p=0.9, rep_penalty≈1.05
代码/数学   T=0.2, top_p=0.95
创意写作    T=1.0~1.2, presence_penalty=0.3~0.6
JSON/函数调用  T=0 + 约束解码（只允许合法 token）
```

---

## 5.3 KV Cache：最重要的一个推理优化

核心就三行（在注意力里把新算的 K/V 拼到缓存上）：

```python
def attention_step(q_new, k_new, v_new, k_cache=None, v_cache=None):
    """q_new:(B,h,1,dh)  k/v_new:(B,h,1,dh)  返回输出与更新后的缓存。"""
    k = torch.cat([k_cache, k_new], dim=2) if k_cache is not None else k_new
    v = torch.cat([v_cache, v_new], dim=2) if v_cache is not None else v_new
    scores = (q_new @ k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)
    w = torch.softmax(scores, dim=-1)                    # (B,h,1,T) —— 只有一行，不再算 T×T
    return (w @ v), k, v
```

自回归生成的两段式写法（prefill 一次算完整段，decode 每次只喂 1 个 token）：

```python
# ---- prefill：把 prompt 一次算完，顺便把 K/V 全部缓存下来 ----
logits, past = model(prompt, use_cache=True)
nxt = logits[:, -1].argmax(-1, keepdim=True)

# ---- decode：每步只喂 1 个 token，位置编号接着走 ----
for _ in range(max_new_tokens):
    pos = torch.full((1, 1), cur_len)                    # ← 位置编号必须连续，否则 RoPE 出错
    x = model.emb(nxt) + model.pos(pos)
    for l in range(LAYERS):
        o, k, v = model._attn(x, l, past[l][0], past[l][1])   # 复用缓存
        ...
    nxt = logits[:, -1].argmax(-1, keepdim=True)
```

脚本的真实输出（生成 64 个 token）：

```
无 cache:     45.9 ms
有 cache:     22.1 ms
加速比  : 2.08×    （生成 64 个 token，前缀 8）两者生成的 token 完全一致: True
[OK] KV Cache 输出 vs 全量重算输出: 最大绝对误差 = 0.000e+00 (tol=0)
```

> 误差为 0 —— KV Cache 是**精确等价**的优化，和量化/剪枝不是一回事。
> 加速比在小模型上只有 2×（Python 循环开销占比大），真实 LLM 上通常 10× 以上。

脚本：`script/05_llm/03_kv_cache.py`

问题：第 t 步的注意力需要 1..t 所有位置的 K/V，每步重算是 O(T²)。

```
K_cache = concat(K_cache, K_new)
V_cache = concat(V_cache, V_new)
attn    = softmax(Q_new · K_cacheᵀ / √d) · V_cache
```

**它是精确等价的优化**（脚本验证了输出完全一致，误差 0），不改变任何数值结果。

代价是显存：

```
KV Cache ∝ 层数 × KV 头数 × head_dim × 序列长度 × batch
```

三条优化主线：

1. **GQA / MQA**：减少 KV 头数 → 显存降到 1/4 ~ 1/32
2. **MLA**（DeepSeek）：把 KV 压成低秩潜向量
3. **PagedAttention**（vLLM）：分页管理，消除碎片

### Prefill vs Decode

```
Prefill（处理输入）：一次算完 → 计算密集，可大 batch
Decode（逐个生成）：每步 1 个 token → 访存密集，权重反复读
```

> 这就是为什么**量化对 decode 加速特别明显**（省的是带宽），而对 prefill 收益较小。

---

## 5.4 RoPE 与长上下文外推

脚本：`script/05_llm/04_rope.py`

RoPE 每个维度 i 有一个频率 `θ_i = 1/base^(i/d)`：

```
小 i → 高频、波长短（几个 token）→ 负责【局部】
大 i → 低频、波长长（上万 token）→ 负责【长程】
```

三种缩放（脚本量化对比）：

| 方案 | 局部分辨率 | 长程相位比 | 代价 |
|---|---|---|---|
| 直接外推 | 1.00 | 8.00× | 超出训练分布 → 分数畸变 |
| 线性插值 PI | 0.125 | 1.00 | 相邻位置难分辨 |
| NTK-Aware | 1.00 | 1.00 | 中间频段仍有畸变 |
| YaRN | 0.90 | 0.90 | + 注意力温度校正，最稳 |

实现风格有两种（**混用会导致输出完全不同，迁移权重时必须对齐**）：

- 相邻两维一组（GPT-J 风格）
- 前后半分组（Llama / HuggingFace 风格）

---

## 5.5 量化

对称 absmax 量化（10 行）+ 分组量化：

```python
def absmax_quantize(w, bits=8, group_size=None):
    qmax = 2 ** (bits - 1) - 1                    # int8 → 127
    if group_size is None:                        # 整张量一个 scale
        scale = np.abs(w).max() / qmax
        q = np.round(w / scale).clip(-qmax - 1, qmax)
        return q.astype(np.int32), np.array([scale])
    w2 = w.reshape(-1, group_size)                # 每组一个 scale（精度大幅提升）
    scales = np.abs(w2).max(axis=1, keepdims=True) / qmax
    q = np.round(w2 / scales).clip(-qmax - 1, qmax)
    return q.astype(np.int32).reshape(w.shape), scales.squeeze()

def dequantize(q, scales, group_size=None):
    if group_size is None:
        return q.astype(np.float32) * scales[0]
    return (q.reshape(-1, group_size).astype(np.float32) * scales[:, None]).reshape(q.shape)
```

整数矩阵乘（INT8 TensorCore 干的事）：

```python
def int8_matmul(W_q, W_scale, X_q, X_scale):
    out_i32 = W_q.astype(np.int64) @ X_q.astype(np.int64)      # int8 × int8 → int32 累加
    return (out_i32 * W_scale * X_scale).astype(np.float32)
```

脚本的真实误差对比（512×256 权重，含 2 个离群值）：

```
group_size=None   输出相对误差=0.103020   额外 scale 开销=0.00B
group_size=128    输出相对误差=0.007488   额外 scale 开销=4.00KB
group_size=64     输出相对误差=0.006521   额外 scale 开销=8.00KB
4-bit NF4 (group=64)  = 0.09497
int8     (group=64)   = 0.00664
```

离群值有多可怕（脚本输出）：

```
含离群值时的 scale  = 0.007244     (max|W| = 0.9200)
去掉离群值后的 scale = 0.000745     → 小了 9.7×
去掉离群值后的量化误差 = 0.01075    （含离群值时是 0.10302）
```

脚本：`script/05_llm/05_quantization.py`

```
对称 absmax： s = max|W| / 127,   W_int8 = round(W / s)
```

实测（512×256 权重矩阵，含 2 个离群值）：

| 方案 | 输出相对误差 |
|---|---|
| int8 per-tensor | 0.103 |
| int8 group=128 | 0.0075 |
| int8 group=64 | 0.0065 |
| 4-bit NF4 group=64 | 0.095 |

关键洞察：

1. **离群值是量化误差的主要来源** → AWQ / SmoothQuant 的核心就是"把离群值平滑掉"
2. **分组量化**用极小的 scale 开销（每组一个 fp32）换来数量级的精度提升
3. **W4A16**（只量化权重）最实用：decode 时省的是**显存带宽**，这才是瓶颈
4. QLoRA：4-bit 底座 + fp16 LoRA → 单卡 48G 微调 65B

选型：

```
显存不够        → GPTQ / AWQ 4-bit
要最高质量      → fp16 / bf16
服务端高吞吐    → fp8（H100）+ 连续批处理
单卡微调        → QLoRA
端侧 / CPU      → GGUF Q4_K_M（llama.cpp）
```

---

## 5.6 FlashAttention

在线 softmax（FlashAttention 的算法核心，15 行）：

```python
def attention_flash(Q, K, V, block_size=64, is_causal=True):
    T, d = Q.shape[0], Q.shape[-1]
    O = torch.zeros_like(Q)
    m = torch.full((T, 1), -float("inf"))        # running max
    l = torch.zeros((T, 1))                      # running sum(exp)

    for j in range(0, T, block_size):            # 遍历 K/V 块，一次只读一块
        Kj, Vj = K[j:j+block_size], V[j:j+block_size]
        Sij = Q @ Kj.T / (d ** 0.5)              # (T, Bj) —— 从来没有 T×T 矩阵
        if is_causal:
            rows = torch.arange(T)[:, None]
            cols = torch.arange(j, j + block_size)[None, :]
            Sij = Sij.masked_fill(cols > rows, float("-inf"))
        m_new = torch.maximum(m, Sij.max(dim=-1, keepdim=True).values)
        correction = torch.exp(m - m_new)        # ← 旧累加量要按新的 max 修正
        Pij = torch.exp(Sij - m_new)
        l = correction * l + Pij.sum(dim=-1, keepdim=True)
        O = correction * O + Pij @ Vj
        m = m_new
    return O / l
```

脚本验证（**精确等价，不是近似**）：

```
[OK] 朴素注意力 vs Flash（在线 softmax）: 最大绝对误差 = 2.980e-07 (tol=1e-05)
```

显存对比（脚本输出，单头 fp16）：

```
序列长度 T        朴素注意力 S+P        Flash (仅 O)      节省倍数
  8192                 256.00MB            512.00KB         512
 32768                   4.00GB              2.00MB        2048
131072                  64.00GB              8.00MB        8192
```

> 反向怎么办：不保存 P，反向时用 O、m、l **重算**分块内的 P。
> 多花 ~30% 算力，省 O(T²) 显存 —— 在 GPU 上重算通常比从 HBM 读更快。

脚本：`script/05_llm/06_flash_attention.py`

两个核心技术：

1. **分块（tiling）**：绝不把 T×T 矩阵写回显存
2. **在线 softmax**：流式维护 running max / running sum

```python
m_new = max(m, block_max)
corr  = exp(m - m_new)                 # 修正因子
l     = corr * l + sum(exp(block - m_new))
O     = corr * O + exp(block - m_new) @ V
```

脚本验证：**朴素与在线 softmax 结果完全一致**（误差 1e-6）→ 它是精确算法，不是近似。

显存对比（d=32 单头 fp16）：

```
T=8192   : 朴素 256MB  vs  Flash 0.5MB   （512×）
T=131072 : 朴素 64GB   vs  Flash 8MB
```

**反向怎么办**：不保存 P（T×T），反向时用 O、m、l **重算**分块内的 P。
这是经典的"用算力换显存"：多花 ~30% 算力，省 O(T²) 显存。
在 GPU 上重算通常比从 HBM 读更快 —— 第 7 章会反复印证这一点。

落地：`F.scaled_dot_product_attention` 会自动选 flash / mem-efficient / math 后端。

---

## 5.7 LoRA / PEFT

```python
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank=4, alpha=8.0):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad_(False)              # 冻结原权重
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)
        self.rank, self.scaling = rank, alpha / rank
        self.A = nn.Parameter(torch.randn(rank, linear.in_features) * 0.02)
        self.B = nn.Parameter(torch.zeros(linear.out_features, rank))   # ← B 必须为 0

    def forward(self, x):
        return self.linear(x) + self.scaling * F.linear(F.linear(x, self.A), self.B)

    def delta_w(self):
        return self.scaling * (self.B @ self.A)

    def merge(self):                                          # 推理前合并 → 零额外开销
        with torch.no_grad():
            self.linear.weight += self.delta_w()
        return self.linear
```

关键点：`B = 0` 保证训练开始时 `ΔW = 0`，不破坏原模型；`scaling = alpha/rank`
让不同 rank 下的"有效学习率"大致可比。

脚本的真实对比（同一任务）：

```
方案             可训练参数        占比      最终 MSE
全量微调              4,192   100.00%     0.13966
仅最后一层            2,080    49.62%     0.21775
LoRA r=4               768    18.32%     0.25183
（基准：不微调 MSE=0.3774）

rank 扫描： r=1 →0.29919   r=2 →0.28076   r=4 →0.25089   r=8 →0.21578   r=16 →0.17668
```

合并后验证（脚本输出）：

```
||ΔW|| = 0.5231   原权重 ||W|| = 3.1840
合并前后输出的最大差异 = 0.00e+00      ← 完全等价，部署时没有额外算子
```

脚本：`script/05_llm/07_lora_peft.py`

核心假设：微调的权重更新 ΔW 是**低秩**的。

```
W' = W + B·A        A:(r×d_in), B:(d_out×r), r ≪ d
y  = W·x + (α/r)·B·(A·x)
初始化：A ~ N(0,σ²)，B = 0  → 训练开始时 ΔW = 0，不破坏原模型
```

rank 越大效果越好（r=1→0.299, r=16→0.177 MSE），但也越接近全量微调的成本。

**推理时可以合并权重**（零额外延迟，这是相比 Adapter 的巨大优势）：

```python
with torch.no_grad():
    linear.weight += (alpha / rank) * (B @ A)      # 合并后就是普通 Linear，零额外开销
```

实战：

- `target_modules`：优先 q_proj / v_proj（原论文），实践中所有线性层都加更好
- 学习率：1e-4 ~ 2e-4（比全量微调高一个量级）
- 数据量大且要大幅改变行为时（继续预训练）→ 全量微调更好

---

## 5.8 推理服务：连续批处理 + PagedAttention

连续批处理的核心循环（每个 step 检查谁完成了，立刻补新请求）：

```python
def simulate_continuous(requests, max_batch=4):
    pending = sorted(requests, key=lambda r: r[0])
    running, done, time = [], [], 0.0
    while pending or running:
        while pending and len(running) < max_batch and pending[0][0] <= time:
            r = pending.pop(0)
            running.append([r, r[2], time])           # (请求, 剩余生成数, 开始时间)
        time += 1
        still = []
        for req, left, start in running:
            left -= 1
            if left <= 0:                        # 这个请求生成完了 → 立刻腾出槽位
                done.append((req, time))
            else:
                still.append([req, left, start])
        running = still                          # 下一轮 while 会补进新请求
    return done, time
```

页表（逻辑块 → 物理块）：

```python
class BlockTable:
    def __init__(self, block_size, num_blocks):
        self.block_size, self.free = block_size, list(range(num_blocks))
        self.table = {}                                  # req_id -> [物理块号]

    def append(self, req_id, used_tokens):
        need = (used_tokens + self.block_size - 1) // self.block_size
        while len(self.table[req_id]) < need:            # 按需申请，不预分配
            self.table[req_id].append(self.free.pop(0))

    def lookup(self, req_id, token_idx):
        return self.table[req_id][token_idx // self.block_size], token_idx % self.block_size
```

脚本的真实输出：

```
静态批处理总耗时 184 步      连续批处理总耗时 120 步   吞吐提升 1.53×

预分配最大长度  : 6.00GB   （利用率 22.4%）
实际需要        : 1.34GB
PagedAttention  : 1.36GB   （利用率 98.7%）
节省显存        : 4.64GB  → 可多跑 341% 的并发
```

脚本：`script/05_llm/08_batching_and_paged_kv.py`

### 连续批处理

静态批处理要等整批跑完（**head-of-line blocking**）；
连续批处理每个请求生成完一个 token 就让位，立刻插入新请求。

### PagedAttention

预分配"最大长度"浪费严重（数字见上一节的脚本输出：24 个请求各预留 2048 长度时，
实际利用率只有 22.4%，换成 PagedAttention 后是 98.7%，同样的卡能多跑 341% 的并发）。

同一个请求的 KV 在物理显存里**不连续** → 注意力 kernel 必须支持"按页表 gather"，
这就是 vLLM 要自己写 CUDA kernel 的原因（第 7 章）。

### 前缀共享（prefix caching）

32 个请求共享 500 token 的 system prompt：

```
不共享 2.6GB  →  共享后 0.6GB   （节省 77%）
```

> 所以"把不变的 system prompt 放前面"是真正有用的工程技巧。

### 生产配置清单

```
框架：vLLM（吞吐）/ TGI（功能全）/ TensorRT-LLM（极致性能）/ llama.cpp（端侧）
参数：max_num_batched_tokens（首 token 延迟）
     gpu_memory_utilization 0.85~0.95
     enable_prefix_caching（多轮对话必开）
     speculative decoding（小模型起草 + 大模型验证，2~3× 加速，不改变输出分布）
```

---

## 5.9 速记卡

```
· 显存 = 权重 + KV Cache + 激活；训练时每参数约 16 字节
· KV Cache ∝ 层数 × KV头数 × head_dim × seq × batch → 长上下文的主要瓶颈
· Prefill 拼算力，Decode 拼带宽 → 量化主要加速 decode
· 采样默认 top_p=0.9 + T=0.7；代码/数学用低温
· FlashAttention 是精确算法：分块 + 在线 softmax，反向用重计算
· 量化的敌人是离群值；分组量化是标配；QLoRA = 4bit 底座 + LoRA
· LoRA 可合并回原权重 → 零推理开销
· 服务端三件套：连续批处理 + PagedAttention + 前缀共享
```

---

## 5.10 真实生态：generate() / 真实 KV Cache / peft

```bash
python3 script/05_llm/09_real_generation_kvcache.py    # transformers，会下载 Qwen2.5-0.5B
python3 script/05_llm/10_real_peft_lora.py             # peft
```

### 真实 generate() 的参数效果（Qwen2.5-0.5B）

```python
out = model.generate(**enc, max_new_tokens=24,
                     do_sample=True, temperature=0.8, top_p=0.9,
                     use_cache=True, pad_token_id=tok.pad_token_id)
```

```
greedy                 → ' is a ________.\nA. noun\nB. verb\nC. adjective\nD. adverb\nAnswer:\n'
temperature=0.7        → ' is on the left. The quick brown fox jumps over the lazy dog. The dog '
temperature=1.5        → ' jumps under the mat. On which is the word ____?\nmat\njumper\n...'
top_p=0.9              → ' is on the left of the quick fox. The fox is on the right of the lazy '
repetition_penalty=1.5 → ' is a ________.\nA: noun\nB:\nAnswer:\n\nC\n\nWhich of these ...'
beam=3                 → ' is lazy, and so are the lazy dogs that follow it. Given the context:'
```

> 温度越高越发散（T=1.5 已经开始跑题）；greedy 最"安全"但也最容易模板化。

### 真实 KV Cache：公式 vs 实测

```python
o = model(**enc, use_cache=True)
cache = o.past_key_values
k0 = cache.layers[0].keys                      # (B, kv_heads, T, head_dim)
total = sum(cache.layers[l].keys.numel() + cache.layers[l].values.numel()
            for l in range(len(cache.layers))) * k0.element_size()
```

```
第 0 层 K 形状: (1, 2, 12, 64)   dtype: torch.float32
层数=24  KV 头数=2  head_dim=64  序列长度=12
实测 KV Cache 总大小 = 288.00KB
公式 2·L·T·kv_heads·d_head·bytes = 288.00KB        ← 完全一致
每 token = 24.00KB
```

use_cache 的真实加速（40 个 token）：

```
use_cache=True :    796.6 ms
use_cache=False:   2900.3 ms      加速比 3.64×
```

### peft 的真实 LoRA

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.05,
                    target_modules=["q_proj", "v_proj"])
model_lora = get_peft_model(model, config)
model_lora.print_trainable_parameters()
```

```
基座: Qwen2.5-0.5B，参数量 = 494.03M
trainable params: 540,672 || all params: 494,573,440 || trainable%: 0.1093

target_modules 选择对参数量的影响：
['q_proj']                                    344,064  (0.070%)
['q_proj', 'v_proj']                          540,672  (0.109%)
['q_proj','k_proj','v_proj','o_proj']       1,081,344  (0.219%)
全部注意力 + FFN                             4,399,104  (0.890%)

adapter 文件大小 = 2130 KB（基座 fp32 ≈ 1.9 GB）
合并前后 logits 最大差异 = 0.00e+00     ← 部署时直接合并，零额外延迟
```

训练时的三个必踩坑：

```python
model.config.use_cache = False                  # ① 训练必须关 KV Cache，否则显存爆
model.enable_input_require_grads()              # ② 配合 gradient_checkpointing
labels[attention_mask == 0] = -100              # ③ padding 不参与 loss
```

---

## 5.11 代码索引

| 脚本 | 内容 |
|---|---|
| `01_token_and_memory_budget.py` | Token 预算、推理/训练显存、算力估算器 |
| `02_sampling_strategies.py` | greedy/T/top-k/top-p/惩罚/beam 全部手写 |
| `03_kv_cache.py` | 手写 KV Cache 推理 + 等价性验证 + 加速比 |
| `04_rope.py` | RoPE 实现 + 三种外推缩放的量化对比 |
| `05_quantization.py` | absmax int8 / 分组量化 / NF4 / 整数 matmul |
| `06_flash_attention.py` | 在线 softmax 实现 + 显存对比 + 重计算说明 |
| `07_lora_peft.py` | 手写 LoRA + rank 对比 + 权重合并 |
| `08_batching_and_paged_kv.py` | 连续批处理模拟 + 页表 + 前缀共享 |

```bash
python3 script/05_llm/03_kv_cache.py     # 推荐先跑这个（会打印加速比）
bash script/run_all.sh 05                # 或跑完本章全部 8 个脚本
```
