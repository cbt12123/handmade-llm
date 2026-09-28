"""
06 · 掩码（Mask）：Transformer 里最容易被写错、也最重要的细节
================================================================
三种掩码：
  ① Padding 掩码 : 让所有 query 忽略 <pad> 位置（否则 pad 会污染注意力）
  ② 因果掩码     : 位置 i 只能看 0..i（自回归生成的前提）
  ③ 自定义掩码   : 前缀 LM（GLM）、文档隔离（防止跨文档注意）、稀疏/滑窗注意力

实现要点：
  · 用 bool（True=可见）比用 -inf 更清晰
  · 加性掩码（additive mask）用 0 / -inf，与 scores 相加
  · 整行全被 mask 掉时 softmax 会出现 nan → 必须保证对角线可见
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import ascii_heatmap, require_torch, section, set_seed, subsection

set_seed(6)
torch = require_torch("06_masking")
F = torch.nn.functional


def softmax_np(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def show(name, mask_bool, title=""):
    """打印 0/1 掩码矩阵（1=可见）"""
    print(f"\n  {name}")
    m = mask_bool.astype(int)
    print("\n".join("    " + " ".join(str(v) for v in row) for row in m))


T = 6
section("1) 三种基础掩码（1 = 可以注意到）")
causal = np.tril(np.ones((T, T))).astype(bool)
show("① 因果掩码（下三角，GPT 用）", causal)
lengths = np.array([4, 2])                                    # 两条样本的真实长度
pad = np.zeros((2, T), dtype=bool)
for i, L in enumerate(lengths):
    pad[i, :L] = True
show("② Padding 掩码（按样本，行=样本，列=key 位置）", pad)

combined = causal[None] & pad[:, None, :]                     # (B, T_q, T_kv)
show("③ 合并后的掩码（第 0 条样本）", (combined[0]).astype(int))
print("\n  合并规则：可见 = 因果可见 AND 不是 pad。广播形状 (T_q,T_kv) & (B,1,T_kv) → (B,T_q,T_kv)")

section("2) 掩码是怎么作用到注意力分数上的")
rng = np.random.default_rng(0)
Q = rng.normal(size=(1, 4, T, 8))       # (B, h, T, dh)
K = rng.normal(size=(1, 4, T, 8))
scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(8)
masked = np.where(combined[:, None, :, :], scores, -1e9)      # 不可见位置填 −1e9
w_nomask = softmax_np(scores[0, 0])
w_mask = softmax_np(masked[0, 0])
print("  未加掩码的注意力权重（第 0 头）：")
print("\n".join("    " + " ".join(f"{v:.3f}" for v in row) for row in w_nomask))
print("\n  加了因果掩码后：")
print("\n".join("    " + " ".join(f"{v:.3f}" for v in row) for row in w_mask))
print("\n  上三角变成 0 —— 未来信息被彻底切断。")

section("3) 陷阱：整行被 mask 掉会 nan")
bad_row = np.zeros((1, T), dtype=bool)                        # 这一行一个 key 都看不到
scores_bad = np.where(bad_row, scores[0, 0], -1e9)
out = softmax_np(scores_bad)
print(f"  全 False 行做 softmax 的结果: {np.round(out[0], 3)}  → nan（−inf − (−inf)）")
print("  解决：确保每个 query 至少能看到自己（对角线永远为 True），")
print("        实践中常把 <bos> 或第一个真实 token 强制设为可见。")

section("4) 前缀掩码 / 文档隔离掩码")
prefix_len = 3                                                 # GLM 的做法：前 3 个 token 双向可见
prefix_mask = np.zeros((T, T), dtype=bool)
prefix_mask[:prefix_len, :prefix_len] = True                   # 前缀内部互相可见（双向）
prefix_mask[prefix_len:, :prefix_len] = True                   # 生成部分能看前缀
prefix_mask[np.arange(prefix_len, T), np.arange(prefix_len, T)] = True
prefix_mask |= np.tril(np.ones((T, T)), k=0).astype(bool) & ~np.eye(T, dtype=bool) \
    & (np.arange(T)[:, None] >= prefix_len) & (np.arange(T)[None, :] >= prefix_len)
show("④ 前缀 LM 掩码（前缀双向 + 后续因果）", prefix_mask)

doc_ids = np.array([0, 0, 0, 1, 1, 1])
doc_mask = (doc_ids[:, None] == doc_ids[None, :]) & causal
show("⑤ 文档隔离掩码（同文档内才互相可见）", doc_mask)
print("""
  用途：把多篇文档打包进同一个 batch 时，防止跨文档"抄答案"。
  现代长上下文训练（如 packing）几乎一定会用这种 block-diagonal 掩码。
""")

section("5) PyTorch 里的两种写法")
B, h, Tq, Tk = 2, 4, 6, 6
scores_t = torch.randn(B, h, Tq, Tk)
bool_mask = torch.tensor(doc_mask).expand(B, h, Tq, Tk)
additive = torch.zeros_like(scores_t).masked_fill(~bool_mask, float("-inf"))
w_bool = torch.softmax(scores_t.masked_fill(~bool_mask, float("-inf")), dim=-1)
w_add = torch.softmax(scores_t + additive, dim=-1)
print(f"  bool masked_fill 与 additive mask 结果最大差异 = {(w_bool - w_add).abs().max().item():.2e}")
print("""
  推荐写法（新版 PyTorch）：
      F.scaled_dot_product_attention(q, k, v, attn_mask=bool_mask, is_causal=True)
  它会自动选择 FlashAttention / MemEfficient / math 三种后端之一。
  注意：is_causal=True 时不要再传因果 attn_mask，否则会报错。
""")

section("6) 掩码与注意力变体的关系")
print("""
  · 滑动窗口注意力（Mistral / Longformer）= 带状掩码：|i − j| ≤ W 才可见
  · 稀疏注意力（BigBird / Longformer）= 局部 + 随机 + 全局 token 掩码
  · 线性注意力 / SSM（Mamba）= 用递归取代掩码，把注意力矩阵"隐式"化（第 6 章）
  一句话：很多"新架构"本质上只是换了一种掩码模式或把它们做成了可学习的。
""")
