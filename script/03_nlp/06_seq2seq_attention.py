"""
06 · Seq2Seq + Attention：通往 Transformer 的最后一级台阶
==========================================================
原始 Seq2Seq（2014）：把整句压成一个向量 c，再解码 → 长句信息塞不下（瓶颈问题）
Bahdanau Attention（2015）：解码每一步都"回头看"一遍源句，按相关性加权求和
    score(h_dec, h_enc) = vᵀ·tanh(W₁·h_dec + W₂·h_enc)    加性注意力
    α = softmax(score)          context = Σ αᵢ·h_enc_i

本脚本：
  1. 手写加性 / 点积 / 缩放点积 三种打分函数（第三种就是 Transformer 用的那个）
  2. 在"英→中"玩具词表上跑一遍对齐，用 ASCII 热力图看注意力落在哪
  3. 指出 RNN+Attention 剩下的唯一问题：串行 → 下一章 Transformer 直接砍掉 RNN
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import ascii_heatmap, section, set_seed, subsection

set_seed(15)

# ----------------------------------------------------------------------------------
# 玩具"翻译"任务：为了让注意力有意义，我们手工构造"语义对齐"的向量
# ----------------------------------------------------------------------------------
SRC = ["i", "love", "machine", "learning"]
TGT = ["我", "爱", "机器", "学习"]
# 对应关系：i↔我, love↔爱, machine↔机器, learning↔学习
D = 16
rng = np.random.default_rng(0)
base = rng.normal(size=(4, D))
src_vecs = base + rng.normal(0, 0.15, (4, D))                    # 源语言向量
tgt_vecs = base + rng.normal(0, 0.15, (4, D))                    # 目标语言向量（与源同构）
print(f"源句: {' '.join(SRC)}")
print(f"目标句: {' '.join(TGT)}")


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


section("1) 三种打分函数")
W1 = rng.normal(0, 0.2, (D, D))
W2 = rng.normal(0, 0.2, (D, D))
v = rng.normal(0, 0.2, D)


def additive_scores(dec_h, enc_hs):
    """Bahdanau：加性注意力，先变换再过 tanh 再点乘 v"""
    proj = enc_hs @ W2.T                            # (Ts, D)
    return np.tanh(dec_h @ W1.T + proj) @ v         # (Ts,)


def dot_scores(dec_h, enc_hs):
    """Luong：直接点积，无参数"""
    return enc_hs @ dec_h


def scaled_dot_scores(dec_h, enc_hs):
    """Transformer：点积 / √d —— 除以 √d 防止点积方差过大导致 softmax 饱和"""
    return enc_hs @ dec_h / np.sqrt(D)


dec_h = tgt_vecs[2]                                  # 解码第 3 个词"机器"时的隐藏状态
for name, fn in [("加性 additive", additive_scores), ("点积 dot", dot_scores), ("缩放点积 scaled", scaled_dot_scores)]:
    s = fn(dec_h, src_vecs)
    a = softmax(s)
    top = int(np.argmax(a))
    print(f"  {name:<16} α = {np.round(a, 3)}   最关注: {SRC[top]}")

print(f"\n  解码到『{TGT[2]}』时，注意力正确地指向了『{SRC[2]}』—— 这就是对齐（alignment）。")

section("2) 完整解码过程：一步步生成，看对齐矩阵")
def decode_step(dec_h, enc_hs, score_fn=scaled_dot_scores):
    s = score_fn(dec_h, enc_hs)
    a = softmax(s)
    context = a @ enc_hs                             # 加权求和得到上下文向量
    return a, context


align = []
for i, dh in enumerate(tgt_vecs):
    a, ctx = decode_step(dh, src_vecs)
    align.append(a)
    print(f"  生成 {TGT[i]}  ←  注意力 {np.round(a, 3)}  上下文向量范数 {np.linalg.norm(ctx):.3f}")
align = np.array(align)
print(ascii_heatmap(align, width=48, height=12, title="\n对齐矩阵（行=目标词, 列=源词）："))
print("  对角线亮 = 翻译对齐正确；真实翻译里会出现多对多、语序调换（如英日的动词后置）。")

section("3) 为什么点积要除以 √d？")
for d in [4, 16, 64, 256]:
    q = rng.normal(size=d)
    k = rng.normal(size=d)
    print(f"  d={d:<5} q·k = {float(q @ k):>8.2f}   q·k/√d = {float(q @ k / np.sqrt(d)):>8.2f}")
print("""
  点积的方差随 d 线性增长 → d 大时 softmax 输入量级很大 → 梯度接近 0（饱和）
  除以 √d 把方差拉回 1 —— 这一行除法就是 Transformer 论文里最关键的一句话。
""")

section("4) 编码器-解码器 vs 纯注意力")
print("""
  RNN + Attention 仍然存在的问题：
    ① 编码器必须按时间步串行 → 训练/推理都慢
    ② 源句内部词语之间的关系也要靠递归传递（同样是长程依赖问题）
  Transformer 的解法（下一章）：
    把 RNN 全部删掉，只留下三种注意力
      · 编码器自注意力（源句内部互相看）
      · 解码器掩码自注意力（只看已生成的部分）
      · 交叉注意力（解码器看编码器 —— 就是本节的这套东西）
  一句话：Attention is All You Need = "把本节这套机制从『每秒一步』变成『一次矩阵乘法』"。
""")

section("5) 从本节到下一章的映射表")
print("""
  本节概念                      Transformer 里的对应物
  ---------------------------------------------------------------
  h_enc_i（编码器隐藏状态）  →  Key / Value
  h_dec（解码器隐藏状态）    →  Query
  score(q, k)                →  QKᵀ / √d
  Σαᵢ·h_i                    →  softmax(QKᵀ/√d)·V
  单个注意力头               →  多头注意力（并行做 h 次，再拼起来）
""")
