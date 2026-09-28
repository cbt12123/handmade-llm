"""
04 · 手写多层感知机（MLP）与反向传播
=====================================
这是整个教程最重要的一节：后面的 CNN / Transformer / LLM，
本质都是"更复杂的计算图 + 同一套链式法则"。

网络：  输入(2) → Linear → tanh → Linear → tanh → Linear(1) → sigmoid
损失：  二分类交叉熵

反向传播的通用模式（请背下来）：
    前向时缓存每一层的输入 x 和激活输出 a
    反向时从后往前：dW = xᵀ·dout , dx = dout·Wᵀ , 中间乘上激活函数的导数
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import ascii_heatmap, check_close, grad_check, section, set_seed, subsection

set_seed(11)

# ----------------------------------------------------------------------------------
# 数据：两个半月（线性不可分，逼着网络必须用隐藏层）
# ----------------------------------------------------------------------------------
def two_moons(n_per=120, noise=0.12):
    ang = np.linspace(0, np.pi, n_per)
    x1 = np.c_[np.cos(ang), np.sin(ang)] + np.random.randn(n_per, 2) * noise
    x2 = np.c_[1 - np.cos(ang), 0.5 - np.sin(ang)] + np.random.randn(n_per, 2) * noise
    return np.vstack([x1, x2]), np.hstack([np.zeros(n_per), np.ones(n_per)])


X, y = two_moons()
X = (X - X.mean(0)) / X.std(0)
n, in_dim = X.shape
H1, H2 = 16, 16

# ----------------------------------------------------------------------------------
# 参数初始化（Xavier：让各层输出方差大致一致，避免一开始就饱和）
# ----------------------------------------------------------------------------------
rng = np.random.default_rng(0)


def xavier(fan_in, fan_out):
    return rng.normal(0, np.sqrt(2.0 / (fan_in + fan_out)), (fan_in, fan_out))


W = [xavier(in_dim, H1), xavier(H1, H2), xavier(H2, 1)]
B = [np.zeros(H1), np.zeros(H2), np.zeros(1)]


def tanh(z):
    return np.tanh(z)


def dtanh(a):
    return 1.0 - a**2                        # tanh' = 1 − tanh²，用激活值就能算


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ----------------------------------------------------------------------------------
# 前向：缓存中间结果；反向：链式法则
# ----------------------------------------------------------------------------------
def forward(X_in, W, B, cache=None):
    """三层 MLP 前向。cache 用来保存反向传播需要的中间量。"""
    z1 = X_in @ W[0] + B[0]
    a1 = tanh(z1)
    z2 = a1 @ W[1] + B[1]
    a2 = tanh(z2)
    z3 = a2 @ W[2] + B[2]
    p = sigmoid(z3).ravel()                # 拉平成 (n,)，避免和 1-D 的 y 广播成 (n,n)
    if cache is not None:
        cache.update(X=X_in, z1=z1, a1=a1, z2=z2, a2=a2, p=p)
    return p


def loss_fn(W, B):
    p = forward(X, W, B)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def backward(W, B, cache):
    """返回与 W、B 同形的梯度列表。"""
    m = X.shape[0]
    p = cache["p"]
    dz3 = (p - y).reshape(-1, 1) / m                       # 输出层：sigmoid+BCE 的梯度
    dW3 = cache["a2"].T @ dz3
    dB3 = dz3.sum(0)

    da2 = dz3 @ W[2].T
    dz2 = da2 * dtanh(cache["a2"])                         # 乘激活导数
    dW2 = cache["a1"].T @ dz2
    dB2 = dz2.sum(0)

    da1 = dz2 @ W[1].T
    dz1 = da1 * dtanh(cache["a1"])
    dW1 = X.T @ dz1
    dB1 = dz1.sum(0)
    return [dW1, dW2, dW3], [dB1, dB2, dB3]


section("1) 梯度检查：手写的反向传播正确吗？（这是必做步骤）")
cache = {}
_ = forward(X, W, B, cache)
dW, dB = backward(W, B, cache)
flat_W = np.concatenate([w.ravel() for w in W])


def loss_from_flat(flat):                                   # 把参数摊平，便于有限差分
    idx = 0
    Wl = []
    for w in W:
        size = w.size
        Wl.append(flat[idx:idx + size].reshape(w.shape))
        idx += size
    return loss_fn(Wl, B)


grad_check(loss_from_flat, flat_W, np.concatenate([g.ravel() for g in dW]),
           eps=1e-6, tol=1e-6, name="dL/dW（全部权重）")

section("2) 训练 3000 步（全批量梯度下降，lr=0.5）")
lr = 0.5
for step in range(3001):
    cache = {}
    p = forward(X, W, B, cache)
    dW, dB = backward(W, B, cache)
    for i in range(3):
        W[i] -= lr * dW[i]
        B[i] -= lr * dB[i]
    if step % 500 == 0:
        acc = float(np.mean((p.ravel() >= 0.5) == y))
        print(f"  step {step:>5}  loss={loss_fn(W, B):.5f}  acc={acc:.4f}")

p = forward(X, W, B)
print(f"\n最终训练集准确率 = {np.mean((p.ravel() >= .5) == y):.4f}")

section("3) 单样本视角：一次前向到底在算什么")
sample = X[0]
print(f"  输入 x = {np.round(sample, 3)}   (真值 y={y[0]})")
c = {}
out = forward(sample.reshape(1, -1), W, B, c)
print(f"  第 1 层激活 a1 的形状 = {c['a1'].shape}，前 8 个值 = {np.round(c['a1'].ravel()[:8], 3)}")
print(f"  输出概率 p = {out.item():.4f}")
print(ascii_heatmap(c["a1"].reshape(4, 4), width=32, height=8, title="  第 1 层激活（16 维重排成 4×4）："))

subsection("4) 参数量：一个 2-16-16-1 的网络有多少参数？")
total = sum(w.size for w in W) + sum(b.size for b in B)
print(f"  W 形状: {[w.shape for w in W]}")
print(f"  总参数 = {total}  → 注意：矩阵乘法的参数量 = fan_in × fan_out，这就是 LLM 参数量的主要来源")

section("5) 结论速记")
print("""
  · 反向传播就是"从后往前反复用 dW = xᵀ·dout"，没有任何神秘之处
  · 激活函数的导数只需用激活值本身表达（tanh'=1−a²、sigmoid'=a(1−a)），所以前向必须缓存
  · 线性层叠起来还是线性 → 必须在层之间插非线性，否则再深也等价于一层
  · 把这里的"层"换成 Self-Attention，你就得到了 Transformer
""")
