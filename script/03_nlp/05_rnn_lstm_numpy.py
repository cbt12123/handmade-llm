"""
05 · 手写 RNN / LSTM 前向，并用数值实验看"梯度消失"
=====================================================
RNN：  h_t = tanh(W_xh·x_t + W_hh·h_{t−1} + b)
LSTM： 用"细胞状态 c_t"这条加法高速公路来传递长期信息
       f_t=σ(...)  遗忘门
       i_t=σ(...)  输入门
       g_t=tanh(...) 候选
       o_t=σ(...)  输出门
       c_t = f_t⊙c_{t−1} + i_t⊙g_t          ← 关键：加法，不是反复乘矩阵
       h_t = o_t ⊙ tanh(c_t)

本脚本用真实的梯度回传数值（不是比喻）证明：
  RNN 的梯度随时间步指数衰减；LSTM 的 c 路径梯度只乘 f_t（≈1），可以走很远。
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import section, set_seed, subsection

set_seed(12)

T, DX, DH = 30, 8, 16
rng = np.random.default_rng(0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ----------------------------------------------------------------------------------
section("1) RNN 前向")
W_xh = rng.normal(0, 0.3, (DH, DX))
W_hh = rng.normal(0, 0.3, (DH, DH))
# 把 W_hh 的谱半径调到 ~0.9（典型初始化下会小于 1）
W_hh = W_hh / np.abs(np.linalg.eigvals(W_hh)).max() * 0.9
b_h = np.zeros(DH)

X = rng.normal(size=(T, DX))
h = np.zeros(DH)
hs, a_s = [], []
for t in range(T):
    a = W_xh @ X[t] + W_hh @ h + b_h
    h = np.tanh(a)
    a_s.append(a)
    hs.append(h.copy())
hs = np.array(hs)
print(f"  隐藏状态序列形状 {hs.shape}，最后 5 步的 ||h_t|| = {np.round(np.linalg.norm(hs[-5:], axis=1), 4)}")

section("2) RNN 的梯度回传：||∂L/∂h_t|| 随时间步指数衰减")
def dtanh(a):
    return 1.0 - np.tanh(a) ** 2


g = 2.0 * hs[-1].copy()                     # 取 L = ||h_T||², 则 ∂L/∂h_T = 2h_T
grads_rnn = [np.linalg.norm(g)]
for t in range(T - 2, -1, -1):
    g = W_hh.T @ (dtanh(np.array(a_s[t + 1])) * g)   # ∂L/∂h_t = W_hhᵀ·diag(tanh'(a_{t+1}))·∂L/∂h_{t+1}
    grads_rnn.append(np.linalg.norm(g))
grads_rnn = np.array(grads_rnn[::-1])       # 按时间从 0 到 T−1 排列
print("  时间步 t :", " ".join(f"{t:>3}" for t in range(0, T, 3)))
print("  梯度范数 :", " ".join(f"{grads_rnn[t]:>6.2e}" for t in range(0, T, 3)))
print(f"  t=0 的梯度 / t=T−1 的梯度 = {grads_rnn[0] / grads_rnn[-1]:.3e}")
print("  → 30 步之前的 token 对当前损失的贡献几乎为 0，这就是『长程依赖学不到』的根源。")

# ----------------------------------------------------------------------------------
section("3) LSTM 前向（单步展开）")
W_x = rng.normal(0, 0.3, (4 * DH, DX))
W_h = rng.normal(0, 0.3, (4 * DH, DH))
b = np.zeros(4 * DH)
b[DH:2 * DH] = 1.0                          # 遗忘门偏置初始化为 1（常用 trick：先别忘）
c = np.zeros(DH)
h = np.zeros(DH)
fs, cs, hs_l = [], [], []
for t in range(T):
    gates = W_x @ X[t] + W_h @ h + b
    i = sigmoid(gates[:DH])
    f = sigmoid(gates[DH:2 * DH])
    g_gate = np.tanh(gates[2 * DH:3 * DH])
    o = sigmoid(gates[3 * DH:])
    c = f * c + i * g_gate                  # ← 加法更新
    h = o * np.tanh(c)
    fs.append(f.copy())
    cs.append(c.copy())
    hs_l.append(h.copy())
fs, cs = np.array(fs), np.array(cs)
print(f"  最后 5 步的 ||c_t|| = {np.round(np.linalg.norm(cs[-5:], axis=1), 4)}")
print(f"  遗忘门 f_t 的均值 = {fs.mean():.3f}（接近 1 表示信息能一路传下去）")

section("4) LSTM 的 c 路径梯度：只乘遗忘门")
g_c = 2.0 * cs[-1].copy()                   # 取 L = ||c_T||²
grads_lstm = [np.linalg.norm(g_c)]
for t in range(T - 2, -1, -1):
    g_c = fs[t + 1] * g_c                   # ∂L/∂c_t = f_{t+1} ⊙ ∂L/∂c_{t+1}
    grads_lstm.append(np.linalg.norm(g_c))
grads_lstm = np.array(grads_lstm[::-1])
print("  时间步 t :", " ".join(f"{t:>3}" for t in range(0, T, 3)))
print("  梯度范数 :", " ".join(f"{grads_lstm[t]:>6.2e}" for t in range(0, T, 3)))
print(f"  t=0 的梯度 / t=T−1 的梯度 = {grads_lstm[0] / grads_lstm[-1]:.3e}")

section("5) 两者对比")
print(f"{'模型':<8}{'t=0 处梯度 / t=T−1 处梯度':>28}")
print(f"{'RNN':<8}{grads_rnn[0] / grads_rnn[-1]:>28.3e}")
print(f"{'LSTM':<8}{grads_lstm[0] / grads_lstm[-1]:>28.3e}")
print("""
  · RNN 每一步都要乘一次 W_hh（谱半径 < 1）→ 指数衰减
  · LSTM 的 c 路径是 y = f·y + input 的加法形式 → 梯度只要乘 f（门控是"每步重新决定的"）
  · 但 LSTM 并没有彻底解决长程依赖：它仍然要串行跑 T 步，无法并行 —— 这个缺陷
    直接催生了 Transformer（第 4 章）：把"沿时间的递归"换成"一次矩阵乘法的注意力"。
""")

section("6) 复杂度对比（为什么 RNN 被淘汰）")
print(f"""
  序列长度 T={T}，隐藏维度 d={DH}
    RNN/LSTM : 时间 O(T·d²) 且【必须串行】；GPU 上 T 步无法并行 → 训练慢
    Transformer: 时间 O(T²·d)  但【一步矩阵乘法】→ 全部位置并行，GPU 友好
  当 T 不太大（几千）时，O(T²) 的常数代价远小于串行带来的损失 —— 于是注意力赢了。
""")
