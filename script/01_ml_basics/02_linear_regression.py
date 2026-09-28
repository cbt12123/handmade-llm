"""
02 · 线性回归：机器学习界的 "Hello World"
=========================================
模型：ŷ = Xw + b          损失：MSE = mean((ŷ − y)²)
梯度：dL/dw = (2/n)·Xᵀ(ŷ − y)，  dL/db = (2/n)·sum(ŷ − y)

对比三种求法：
  A. 正规方程（闭式解）      —— 一步到位，但 n 大时 O(n·d²) + 求逆很贵
  B. 全批量梯度下降          —— 循环版本，和深度学习完全一致
  C. 小批量 SGD              —— 大模型训练真正用的形式
另外演示：特征缩放（标准化）为什么是"必需"而不是"可选"。
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import check_close, section, set_seed, subsection

set_seed(7)

# ----------------------------------------------------------------------------------
# 造数据：y = 3·x₁ − 2·x₂ + 0.5·x₃ + 4 + 噪声
# ----------------------------------------------------------------------------------
N, D = 300, 3
X_raw = np.random.randn(N, D) * np.array([1.0, 50.0, 0.01]) + np.array([0.0, 100.0, 0.0])
W_TRUE = np.array([3.0, -2.0, 0.5])
B_TRUE = 4.0
y = X_raw @ W_TRUE + B_TRUE + np.random.randn(N) * 0.5

print(f"数据形状 X={X_raw.shape}, y={y.shape}")
print(f"真实权重 w={W_TRUE}, b={B_TRUE}")


def add_bias(X: np.ndarray) -> np.ndarray:
    """把偏置 b 吸收进权重：在 X 后面拼一列 1，这样只需要维护一个 w。"""
    return np.hstack([X, np.ones((X.shape[0], 1))])


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


section("A. 正规方程（闭式解）：w = (XᵀX)⁻¹Xᵀy")
Xb = add_bias(X_raw)
w_closed = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)     # 比直接 inv 更数值稳定
print(f"  学到 w={np.round(w_closed[:D], 4)}, b={w_closed[D]:.4f}")
print(f"  训练集 MSE = {mse(Xb @ w_closed, y):.6f}")

section("B. 全批量梯度下降（手写循环）")
Xb_norm = add_bias((X_raw - X_raw.mean(0)) / X_raw.std(0))   # 标准化
y_std = (y - y.mean()) / y.std()

w = np.zeros(D + 1)
lr, steps = 0.1, 500
for step in range(steps):
    pred = Xb_norm @ w
    g = (2.0 / N) * (Xb_norm.T @ (pred - y_std))    # MSE 对 w 的梯度
    w -= lr * g
    if step % 100 == 0 or step == steps - 1:
        print(f"  step {step:>4}  loss={mse(pred, y_std):.6f}")
print(f"  标准化空间学到的 w={np.round(w, 4)}")

section("C. 小批量 SGD（batch_size=32）")
w_mb = np.zeros(D + 1)
lr_mb, epochs = 0.05, 6
for ep in range(epochs):
    perm = np.random.permutation(N)
    for start in range(0, N, 32):
        idx = perm[start:start + 32]
        xb, yb = Xb_norm[idx], y_std[idx]
        pred = xb @ w_mb
        w_mb -= lr_mb * (2.0 / len(idx)) * (xb.T @ (pred - yb))
    print(f"  epoch {ep}  loss={mse(Xb_norm @ w_mb, y_std):.6f}")

check_close(w_closed[:D], W_TRUE, tol=0.6, name="闭式解权重 vs 真实权重")
check_close(w_mb, w, tol=0.15, name="小批量 SGD vs 全批量 GD 的解")

section("D. 不做标准化的后果")
Xb_raw = add_bias(X_raw)
w_un = np.zeros(D + 1)
y_raw = y
for step in range(200):
    pred = Xb_raw @ w_un
    g = (2.0 / N) * (Xb_raw.T @ (pred - y_raw))
    w_un -= 1e-4 * g                                  # 只能把 lr 调到极小才不炸
    if not np.isfinite(w_un).all():
        print("  学习率 1e-4 时已经发散（NaN）")
        break
else:
    print(f"  未标准化 + lr=1e-4：200 步后 loss={mse(Xb_raw @ w_un, y_raw):.4f}（还很大）")

print(f"  标准化 +  lr=0.1 ：500 步后 loss={mse(Xb_norm @ w, y_std):.6f}")
print("\n原因：x₂ 的量纲是 100 量级，它对梯度的贡献被放大了几个数量级，")
print("     同一个 lr 要么在 x₂ 方向上发散、要么在 x₃ 方向上走不动。")

section("E. 结论速记")
print("""
  · 线性回归就是"一层没有激活函数的神经网络"，后面的 MLP/Transformer 只是把它叠起来
  · 闭式解 vs 梯度下降：前者 O(n·d²)~O(d³)，后者每步 O(batch·d)；d 很大时只能用后者
  · 特征标准化几乎总是必需的；Transformer 里的 LayerNorm/RMSNorm 干的是同一件事的"动态版"
""")
