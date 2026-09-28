# 第 4 章 · Transformer：一次矩阵乘法取代递归

> 配套代码：`script/04_transformer/`
> 本章从"手算注意力"开始，到一个真正训练出来的 Transformer 结束。

---

## 4.1 缩放点积注意力

全部代码（就是这四行 + 一个 softmax）：

```python
import numpy as np

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))       # 减最大值，数值稳定
    return e / e.sum(axis=axis, keepdims=True)

def attention(Q, K, V, mask=None):
    """Q:(Tq,d) K:(Tk,d) V:(Tk,dv) → out:(Tq,dv), weights:(Tq,Tk)"""
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)                       # ① 打分  ② 缩放
    if mask is not None:
        scores = np.where(mask, scores, -1e9)             # 用 −1e9 代替 −inf，避免 nan
    weights = softmax(scores, axis=-1)                    # ③ 归一化成"注意力分配"
    return weights @ V, weights                           # ④ 加权聚合
```

逐行验证输出确实等于 V 的加权平均（脚本输出）：

```
[OK] 第 2 个输出 = Σⱼ αᵢⱼ·Vⱼ: 最大绝对误差 = 1.110e-16 (tol=1e-12)
```

自注意力 vs 交叉注意力（唯一区别是 K/V 从哪来）：

```python
out_self,  w_self  = attention(X @ W_Q, X  @ W_K, X  @ W_V)     # K/V 来自自身
out_cross, w_cross = attention(X @ W_Q, X2 @ W_K, X2 @ W_V)     # K/V 来自另一句话
```

复杂度（后面所有优化的动机都在这里）：

```python
for t in [512, 2048, 8192, 32768, 131072]:
    print(t, "注意力矩阵 fp16 显存 ≈", t * t * 2 / 1024**2, "MB（单头单样本）")
```

脚本：`script/04_transformer/01_attention_numpy.py`

两种用法：

- **自注意力**：Q/K/V 都来自同一个序列（编码器、GPT）
- **交叉注意力**：Q 来自解码器，K/V 来自编码器（翻译模型、第 3 章的 Seq2Seq）

复杂度（这是后面所有优化动机的根源）：

```
时间 O(T²·d)     显存 O(T²)
T=8192 时单头 fp16 的注意力矩阵就有 128MB —— 层数 × 头数 × batch 之后直接爆显存
```

---

## 4.2 多头注意力

分头与合并（两个 reshape 就够）：

```python
def split_heads(x, h):                 # (T, d_model) → (h, T, d_head)
    T, D = x.shape
    return x.reshape(T, h, D // h).transpose(1, 0, 2)

def merge_heads(x):                    # (h, T, d_head) → (T, d_model)
    h, T, dh = x.shape
    return x.transpose(1, 0, 2).reshape(T, h * dh)

class MultiHeadAttention:
    def __call__(self, X, mask=None, return_weights=False):
        Q, K, V = split_heads(X @ self.W_Q, self.h), split_heads(X @ self.W_K, self.h), \
                  split_heads(X @ self.W_V, self.h)
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.d_head)     # (h, T, T)
        w = softmax(scores, axis=-1)
        ctx = w @ V                                                  # (h, T, d_head)
        out = merge_heads(ctx) @ self.W_O                            # ← W_O 做跨头混合
        return (out, w) if return_weights else out
```

参数量与计算量（脚本输出）：

```
MHA 参数量 = 4·d_model² = 16384（与头数无关！）
计算量     = O(T²·d_model)：分头只是把 d_model 拆成 h 份，总量不变
```

对照 PyTorch（把权重拷进去保证是同一个函数）：

```python
m = nn.MultiheadAttention(D, H, batch_first=True, bias=False)
with torch.no_grad():
    m.in_proj_weight.copy_(torch.cat([W_Q.T, W_K.T, W_V.T]))     # 注意 torch 的权重是转置存的
    m.out_proj.weight.copy_(torch.tensor(W_O.T))
# [OK] 手写 MHA vs torch nn.MultiheadAttention: 误差 ≈ 1e-4
```

脚本：`script/04_transformer/02_multi_head_attention.py`

```
head_i = Attention(X·W_Qⁱ, X·W_Kⁱ, X·W_Vⁱ)      d_head = d_model / h
MHA(X) = Concat(head_1, ..., head_h) · W_O
```

**为什么"多头"几乎是免费的**：

```
参数量 = 4·d_model²（与头数无关）
计算量 = O(T²·d_model)（把 d 拆成 h 份，总量不变）
```

`W_O` 的作用：拼接后做一次线性混合，否则各头之间永远不交流。
典型配置：Llama-3-8B = 32 头 × 128 维 = 4096。

训练后不同头会自然分工（脚本 07 会看到）：

```
头 0：前一个词（局部语法）   头 1：句首 <cls>（全局聚合）
头 2：同类型词（共指/搭配）  头 3：标点与分隔符
```

---

## 4.3 位置编码

正弦位置编码（12 行）：

```python
def sinusoidal_pe(max_len, d_model):
    pe = np.zeros((max_len, d_model))
    pos = np.arange(max_len)[:, None]
    div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(pos * div)                  # 偶数维 sin
    pe[:, 1::2] = np.cos(pos * div)                  # 奇数维 cos
    return pe
```

它的关键性质（脚本验证：用线性变换从 PE(pos) 预测 PE(pos+5)，最大误差 **0.0000**）：

```python
A = np.linalg.lstsq(pe[:L-5], pe[5:], rcond=None)[0]      # 相对位置关系是线性的
print(np.abs(pe[:L-5] @ A - pe[5:]).max())                # → 0.0000
```

RoPE（现代标准，10 行）：

```python
def rope(x, pos, theta=10000.0):
    """把相邻两维看成复平面上的一个点，旋转 pos·θ_i 弧度。"""
    d = x.shape[-1]; half = d // 2
    freqs = 1.0 / (theta ** (np.arange(0, half) / half))
    cos, sin = np.cos(pos * freqs), np.sin(pos * freqs)
    x1, x2 = x[..., :half], x[..., half:]
    out = np.empty_like(x)
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:] = x2 * cos + x1 * sin
    return out
```

相对位置性质（脚本真实输出）：

```
m= 0, n= 0: <R_m q, R_n k>=-2.858638    <R_0 q, R_{n-m} k>=-2.858638   差=0.00e+00
m= 3, n= 7: <R_m q, R_n k>=-9.053564    <R_0 q, R_{n-m} k>=-9.053564   差=3.55e-15
m= 5, n=13: <R_m q, R_n k>=+0.157162    <R_0 q, R_{n-m} k>=+0.157162   差=4.44e-16
```

> 两者完全相等 → 相对位置信息被编码进了 Q·K 的内积，不需要额外的位置向量。

脚本：`script/04_transformer/03_positional_encoding.py`

注意力是**集合运算**（对输入顺序置换等变），必须显式注入位置信息。

| 方案 | 公式 | 外推 | 代表 |
|---|---|---|---|
| 正弦 PE | `PE[pos,2i]=sin(pos/10000^(2i/d))` | 可以 | 原版 Transformer |
| 可学习绝对位置 | `nn.Embedding(max_len, d)` | 不行 | BERT / GPT-2 |
| **RoPE** | 对 Q/K 做旋转 | 较好 | **Llama / Qwen / Mistral** |
| ALiBi | 分数上减距离惩罚 | 好 | BLOOM / MPT |

RoPE 的核心性质（脚本用数值验证，误差 ~1e-7）：

```
⟨R_m·q, R_n·k⟩ = ⟨R_0·q, R_{n−m}·k⟩      只依赖相对距离
```

**长上下文外推**（第 5 章第 4 节展开）：

| 方案 | 局部分辨率 | 长程相位比 | 说明 |
|---|---|---|---|
| 直接外推 | 1.00 | 8× | 训练时没见过这么大的相位 → 注意力分数畸变 |
| 线性插值 PI | 0.125 | 1.0 | 相位压回训练分布，但相邻位置变难分辨 |
| NTK-Aware | 1.00 | ~1.0 | 只压低频、保留高频 → 两头都顾上 |
| YaRN | ~0.9 | ~0.9 | NTK + 温度校正，实测最稳 |

---

## 4.4 Transformer Block

```python
import torch.nn as nn

class TransformerBlock(nn.Module):
    """Pre-LN：先归一化再进子层，残差直连（现代 LLM 通用）。"""
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1, cross_attn=False):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.ff   = FeedForward(d_model, d_ff, dropout)
        if cross_attn:
            self.ln_cross = nn.LayerNorm(d_model)
            self.cross    = MultiHeadAttention(d_model, num_heads, dropout)

    def forward(self, x, memory=None, mask=None, is_causal=False):
        x = x + self.attn(self.ln1(x), mask=mask, is_causal=is_causal)     # 自注意力
        if self.cross is not None and memory is not None:
            x = x + self.cross(self.ln_cross(x), x_kv=memory)              # 交叉注意力
        x = x + self.ff(self.ln2(x))                                       # FFN
        return x
```

参数量拆解（脚本输出，d_model=64）：

```
注意力部分 W_qkv + W_o = 16384
FFN 部分 两个 Linear  = 32768   ← FFN 占大头（约 2/3）
单层 Encoder 参数量 = 49.73K
单层 Decoder 参数量 = 66.24K（多了交叉注意力）
规律：FFN = 2·d·4d = 8d²，注意力 = 4d² → 总约 12d²/层
```

验证因果性（**必须在 eval 模式下做**，否则 dropout 会让两次前向不同）：

```python
layer.eval()
out_full   = layer(x1, is_causal=True)
out_prefix = layer(x1[:, :3], is_causal=True)
print((out_full[0, :3] - out_prefix[0]).abs().max().item())     # → 2.38e-07（浮点误差量级）
```

> 我第一次写这段代码时忘了 `layer.eval()`，结果差异是 **0.81**（看起来像"因果掩码失效"）。
> 差一点点就得出完全错误的结论 —— 做这类验证一定要先关 dropout。

脚本：`script/04_transformer/04_transformer_block.py`

现代结构是 **Pre-LN**（先归一化再进子层）：

```python
x = x + Attention(LayerNorm(x))     # 残差直连
x = x + FFN(LayerNorm(x))
```

参数量拆解（d_model=64）：

```
注意力 W_qkv + W_o = 4d²
FFN 两个 Linear    = 8d²   ← 占大头（约 2/3）
```

估算 LLM 参数：**L 层 × 12d²**（Llama-7B：32 × 12 × 4096² ≈ 6.4B ✔）

### 原版 vs 现代的 5 个差异（很重要）

| 组件 | 原版 (2017) | 现代 |
|---|---|---|
| 归一化位置 | Post-LN | **Pre-LN**（训练更稳） |
| 归一化 | LayerNorm | **RMSNorm**（Llama） |
| 激活 | ReLU | GELU → **SwiGLU** |
| 位置 | 正弦 PE | **RoPE / ALiBi** |
| 注意力实现 | 朴素 softmax(QKᵀ)V | **FlashAttention** |

---

## 4.5 掩码：最容易写错的部分

```python
import numpy as np

causal = np.tril(np.ones((T, T))).astype(bool)          # ① 因果（下三角）
lengths = np.array([4, 2])                              # ② 每条样本的真实长度
pad = np.zeros((2, T), dtype=bool)
for i, L in enumerate(lengths):
    pad[i, :L] = True

combined = causal[None] & pad[:, None, :]               # ③ 合并：可见 = 因果可见 AND 不是 pad
scores = np.where(combined[:, None, :, :], scores, -1e9)  # 广播到 (B, h, Tq, Tk)
```

脚本打印的三种掩码（1 = 可以注意到）：

```
① 因果掩码            ② Padding 掩码（行=样本）   ③ 合并后（第 0 条样本）
  1 0 0 0 0 0            1 1 1 1 0 0               1 0 0 0 0 0
  1 1 0 0 0 0            1 1 0 0 0 0               1 1 0 0 0 0
  1 1 1 0 0 0                                      1 1 1 0 0 0
  1 1 1 1 0 0                                      1 1 1 1 0 0
  1 1 1 1 1 0                                      1 1 1 1 0 0
  1 1 1 1 1 1                                      1 1 1 1 0 0   ← 第 4 列之后全是 pad
```

nan 陷阱（务必自己跑一遍看看）：

```python
bad = np.zeros((1, T), dtype=bool)                      # 这一行一个 key 都看不到
out = softmax_np(np.where(bad, scores[0, 0], -1e9))
print(out)                                              # → nan（−inf − (−inf)）
```

文档隔离掩码（packing 训练必用，两行）：

```python
doc_ids  = np.array([0, 0, 0, 1, 1, 1])
doc_mask = (doc_ids[:, None] == doc_ids[None, :]) & causal      # block-diagonal
```

脚本：`script/04_transformer/06_masking.py`

| 掩码 | 用途 |
|---|---|
| Padding 掩码 | 让所有 query 忽略 `<pad>` 位置 |
| 因果掩码 | 位置 i 只看 0..i（自回归的前提） |
| 前缀掩码 | 前缀双向 + 后续因果（GLM） |
| 文档隔离 | block-diagonal，防止跨文档"抄答案"（packing 必用） |

实现要点：

```python
# 用 bool（True=可见）比 -inf 清晰；合并规则是 AND
combined = causal[None] & pad[:, None, :]           # 广播成 (B, T_q, T_kv)
scores = scores.masked_fill(~combined, float("-inf"))
```

⚠️ **陷阱**：整行全被 mask 掉时 softmax 会出现 `nan`（`-inf − (-inf)`）。
必须保证对角线可见（或强制第一个 token 可见）。

> 很多"新架构"本质上只是换了一种掩码：滑动窗口 = 带状掩码；稀疏注意力 = 局部+随机+全局。

---

## 4.6 真正训练一个 Transformer

学习率 warmup + 余弦衰减（LLM 训练标配，8 行）：

```python
steps, warmup, total = 800, 40, 800
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)

def lr_lambda(step):
    if step < warmup:
        return step / max(warmup, 1)                     # warmup：从 0 线性升到峰值
    prog = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + np.cos(np.pi * prog))              # 余弦衰减到 0

sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
# 每个 step 末尾：sched.step()   ← 别忘了
```

完整训练步（含 teacher forcing 与梯度裁剪）：

```python
model.train()
x, y = make_batch(64, 8)                                 # y 中"读"的阶段设为 -100，不计 loss
logits = model(x)
loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB),
                       y[:, 1:].reshape(-1), ignore_index=-100)
opt.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
opt.step()
sched.step()
```

脚本的真实输出：

```
step    0  loss=3.3253  token 准确率=0.0488  lr=7.50e-05
step  100  loss=0.0063  token 准确率=1.0000  lr=2.95e-03
step  799  loss=0.0008  token 准确率=1.0000  lr=0.00e+00
[训练耗时] 11.01 s
```

长度外推测试（同一模型，只改输入长度）：

```
k=4    token 准确率=0.3438
k=8    token 准确率=1.0000      ← 训练时用的长度
k=12   token 准确率=0.0898
k=16   token 准确率=0.1074
k=24   token 准确率=0.1497
```

自回归生成（推理时没有 teacher forcing）：

```python
for _ in range(k):
    logits = model(ctx)
    nxt = logits[:, -1].argmax(-1, keepdim=True)
    ctx = torch.cat([ctx, nxt], dim=1)          # 把自己的输出拼回去
```

```
待复制: [16, 12, 2, 9, 16, 16, 1, 5]
模型输出: [16, 12, 2, 9, 16, 16, 1, 5]    是否完全一致: True
```

脚本：`script/04_transformer/05_train_toy_copy.py`、`07_attention_visualization.py`

任务：复制 `[a₁..aₖ]`（05）与**反转序列**（07）。

借这两个小任务把 LLM 预训练的工程技巧过一遍：

| 技巧 | 做法 | 为什么 |
|---|---|---|
| Teacher Forcing | 训练时喂真值 | 稳定，但推理时暴露 bias |
| Warmup | 前 40 步 lr 从 0 线性升到峰值 | 避免 Adam 二阶动量未稳定时带崩 |
| 余弦衰减 | lr 按 cos 降到 0 | 最终 loss 还能再降一截 |
| 梯度裁剪 | `clip_grad_norm_(..., 1.0)` | loss spike 时的保命手段 |
| 混合精度 | bf16 计算 + fp32 主权重 | 省显存、吃 TensorCore |

实测（05 脚本）：

```
step  799  loss=0.0008  token 准确率=1.0000  lr=0.00e+00

长度外推： k=4  →0.34   k=8 →1.00   k=12 →0.09   k=16 →0.11   k=24 →0.15
```

> 训练长度之外明显下降：本模型用的是**可学习绝对位置编码**，没见过那么大的位置索引。
> 换成 RoPE / ALiBi 会好很多。

07 脚本训练"反转"任务后可视化注意力 —— 你会看到清晰的**反对角线**：

```
生成第 0 个 token 时最关注位置 5（权重 0.99）
生成第 1 个 token 时最关注位置 4（权重 1.00）
...
```

这说明模型学到的正是"反转"这个算法，而不是死记硬背。

---

## 4.7 可解释性：注意力图能告诉我们什么

在"反转序列"任务上训练后，取生成阶段的注意力（脚本真实输出）：

```
生成第 0 个 token 时最关注位置 5（权重 0.99）
生成第 1 个 token 时最关注位置 4（权重 1.00）
生成第 2 个 token 时最关注位置 3（权重 1.00）
生成第 3 个 token 时最关注位置 2（权重 0.99）
生成第 4 个 token 时最关注位置 1（权重 1.00）
```

> 清晰的**反对角线** = 模型学到的正是"反转"这个算法，而不是死记硬背。

怎么把注意力权重取出来（手写 block 时记得 detach）：

```python
scores = (q @ k.transpose(-2, -1)) / (self.dh ** 0.5)
scores = scores.masked_fill(~causal, float("-inf"))
w = torch.softmax(scores, dim=-1)
attns.append(w.detach())                       # ← 存下来画图；别让它进计算图
```

可视化（把矩阵画成 ASCII 或 matplotlib）：

```python
import matplotlib.pyplot as plt
fig, axes = plt.subplots(LAYERS, H, figsize=(3.2 * H, 3.2 * LAYERS))
for l in range(LAYERS):
    for head in range(H):
        axes[l, head].imshow(attns[l][0, head].numpy(), cmap="viridis")
plt.tight_layout(); plt.savefig("outputs/07_attention_maps.png", dpi=120)
```

真实 LLM 里被反复发现的模式：

| 模式 | 行为 |
|---|---|
| Induction Head（归纳头） | 看到 `A B ... A` 时去关注前一个 A 后面的 B → **上下文学习的基石** |
| Previous Token Head | 只看前一个 token（局部语法） |
| Duplicate Token Head | 关注与当前 token 相同的历史 token（复制/去重） |
| Attention Sink | 大量注意力集中在第一个 token → 丢掉它模型会崩 |

⚠️ **别过度解读**：注意力权重 ≠ 贡献度（后面的 FFN 可能放大或忽略它）。
更严谨的方法是 activation patching / attribution。

---

## 4.8 速记卡

```
· Attention(Q,K,V) = softmax(QKᵀ/√d)·V    除以 √d 是为了防止 softmax 饱和
· 多头是免费的：参数量 4d²，与头数无关
· 注意力是集合运算 → 必须加位置信息；现代标准是 RoPE
· Pre-LN + 残差 = 训练深网络的前提；现代用 RMSNorm + SwiGLU
· 掩码用 bool；合并 = AND；整行全 mask 会 nan
· 参数量 ≈ 层数 × 12d²（FFN 占 2/3）
· 训练标配：warmup + 余弦衰减 + 梯度裁剪 + 混合精度
· 注意力复杂度 O(T²) 是长上下文一切难题的根源
```

---

## 4.9 真实生态：把手写注意力与真实模型对齐

```bash
python3 script/04_transformer/08_real_hf_model.py     # pip install transformers（需联网下模型）
```

最有价值的一步：**用真实模型的权重跑一遍你手写的注意力**，看是否一致。

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "hf-internal-testing/tiny-random-gpt2",
    attn_implementation="eager")        # ← 只有 eager 后端才会返回注意力权重
model.eval()

out = model(ids, output_attentions=True, output_hidden_states=True)
emb   = out.hidden_states[0][0]         # (T, d)
h_in  = model.transformer.h[0].ln_1(emb)   # ← Pre-LN：注意力输入要先过 LayerNorm
attn_hf = out.attentions[0][0]          # (h, T, T)

# 用真实权重手算（GPT-2 用 Conv1D：weight 形状是 (in, out)，前向写成 x @ W + b）
qkv = h_in @ block.attn.c_attn.weight + block.attn.c_attn.bias
q, k, v = qkv.chunk(3, dim=-1)
scores = (q @ k.transpose(-2, -1)) / (dh ** 0.5)
scores = scores.masked_fill(~causal, float("-inf"))
w = torch.softmax(scores, dim=-1)
```

真实输出：

```
HF 注意力权重形状    : (4, 5, 5)
手写注意力权重形状   : (4, 5, 5)
[OK] 手写注意力 vs HF 内部注意力: 最大绝对误差 = 2.980e-08 (tol=1e-05)
```

KV Cache 的真实形状（每步 +1，且每步只需喂 1 个 token）：

```
step 0: 输入 11 token → 缓存 K 形状 (1, 4, 11, 8)
step 1: 输入  1 token → 缓存 K 形状 (1, 4, 12, 8)
step 2: 输入  1 token → 缓存 K 形状 (1, 4, 13, 8)
step 3: 输入  1 token → 缓存 K 形状 (1, 4, 14, 8)
```

> 现代 transformers 里它是 `DynamicCache`（v4.36+），不再是能直接 `kv[0][0]` 下标的 tuple。

只读配置就能看真实大模型（几 KB，秒开，不用下几十 GB 权重）：

```python
from transformers import AutoConfig
c = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B")
print(c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads)
```

---

## 4.10 代码索引

| 脚本 | 内容 |
|---|---|
| `01_attention_numpy.py` | 手写注意力四步 + 与 torch SDPA 对照 + 复杂度表 |
| `02_multi_head_attention.py` | 手写 MHA + 与 `nn.MultiheadAttention` 对照 |
| `03_positional_encoding.py` | 正弦 / 可学习 / RoPE 三种实现 + 外推对比 |
| `04_transformer_block.py` | 完整 Encoder/Decoder、因果性验证、参数量拆解 |
| `05_train_toy_copy.py` | 训练复制任务 + warmup/余弦/裁剪 + 长度外推测试 |
| `06_masking.py` | padding/因果/前缀/文档隔离四种掩码 + nan 陷阱 |
| `07_attention_visualization.py` | 训练反转任务并可视化注意力（反对角线） |

```bash
python3 script/04_transformer/07_attention_visualization.py   # 推荐先跑这个（能看到注意力图）
bash script/run_all.sh 04                                      # 或跑完本章全部 7 个脚本
```
