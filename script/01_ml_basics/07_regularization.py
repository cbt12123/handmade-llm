"""
07 · 过拟合与正则化：L1 / L2 / Dropout / Early Stopping
========================================================
过拟合 = 模型把训练集的噪声也记住了。四种武器：

  L2 (权重衰减)  : loss + λ·Σw²       → 权重整体变小，函数更平滑
  L1             : loss + λ·Σ|w|      → 产生稀疏解（有些权重直接变 0）
  Dropout        : 训练时随机把一部分神经元置零 → 强迫网络不依赖任何单个特征
  Early Stopping : 验证集误差开始上升就停 → 最直接、最便宜

大模型里的对应关系：AdamW 的 weight_decay = L2；
注意力/MLP 里的 dropout 依然是标配；Early Stopping 在 LLM 上等价于"挑合适的 checkpoint"。
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import section, set_seed, subsection

set_seed(21)


def poly_features(x, degree):
    """把一维 x 升维成 [1, x, x², ..., x^degree]。"""
    return np.vander(x.ravel(), degree + 1, increasing=True)


def make_data(n_train=20, n_test=100, noise=0.25, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n_train + n_test)
    y_true = np.sin(np.pi * x)
    y = y_true + rng.normal(0, noise, x.shape)
    return (x[:n_train], y[:n_train]), (x[n_train:], y_true[n_train:])


(xtr, ytr), (xte, yte_true) = make_data()
DEG = 15
Phi_tr, Phi_te = poly_features(xtr, DEG), poly_features(xte, DEG)

section("1) 15 次多项式 + 仅 20 个训练样本：经典过拟合现场")
for lam, name in [(0.0, "无正则"), (1e-4, "L2 λ=1e-4"), (1e-2, "L2 λ=1e-2"), (1.0, "L2 λ=1.0")]:
    A = Phi_tr.T @ Phi_tr + lam * np.eye(DEG + 1)
    w = np.linalg.solve(A, Phi_tr.T @ ytr)
    train_mse = float(np.mean((Phi_tr @ w - ytr) ** 2))
    test_mse = float(np.mean((Phi_te @ w - yte_true) ** 2))
    print(f"  {name:<12} |w|₂={np.linalg.norm(w):>8.2f}  训练 MSE={train_mse:.4f}  测试 MSE={test_mse:.4f}")

print("\n规律：λ 太小 → 训练误差 0 但测试误差爆炸；λ 太大 → 欠拟合，两个误差都大。")

subsection("2) L1 vs L2：稀疏性对比（用坐标下降近似求解 Lasso）")
def lasso(Phi, y, lam, steps=800):
    """坐标下降解 L1 正则：每个维度单独做软阈值。"""
    n, d = Phi.shape
    w = np.zeros(d)
    col_norm = np.sum(Phi**2, axis=0)
    for _ in range(steps):
        for j in range(d):
            r = y - Phi @ w + Phi[:, j] * w[j]           # 去掉第 j 维贡献后的残差
            rho = Phi[:, j] @ r
            w[j] = np.sign(rho) * max(abs(rho) - lam, 0.0) / (col_norm[j] + 1e-12)
    return w


for lam in [1e-3, 1e-2, 1e-1]:
    w_l1 = lasso(Phi_tr, ytr, lam)
    nonzero = int(np.sum(np.abs(w_l1) > 1e-3))
    print(f"  L1 λ={lam:<7} 非零权重个数 = {nonzero}/{DEG + 1}")
A = Phi_tr.T @ Phi_tr + 1e-2 * np.eye(DEG + 1)
w_l2 = np.linalg.solve(A, Phi_tr.T @ ytr)
print(f"  L2 λ=1e-2   非零权重个数 = {int(np.sum(np.abs(w_l2) > 1e-3))}/{DEG + 1}  （L2 只让权重变小，不置零）")

section("3) Dropout：训练时随机失活，推理时全部保留（缩放 1/(1-p)）")
rng = np.random.default_rng(0)
n, d_in, d_h, d_out = 200, 20, 64, 1
X = rng.normal(size=(n, d_in))
true_w = rng.normal(size=(d_in, 1))
y = X @ true_w + rng.normal(0, 0.1, (n, 1))


def train_mlp(dropout_p=0.0, steps=400, lr=0.05, seed=2):
    r = np.random.default_rng(seed)
    W1 = r.normal(0, 0.1, (d_in, d_h))
    W2 = r.normal(0, 0.1, (d_h, d_out))
    for _ in range(steps):
        # ---- 前向 ----
        h = np.maximum(0, X @ W1)                    # ReLU
        if dropout_p > 0:
            mask = (r.random(h.shape) > dropout_p).astype(float) / (1 - dropout_p)
            h = h * mask                             # inverted dropout：训练时放大，推理时不用改
        pred = h @ W2
        # ---- 反向 ----
        g = 2.0 * (pred - y) / n
        dW2 = h.T @ g
        dh = g @ W2.T
        if dropout_p > 0:
            dh = dh * mask
        dh[h <= 0] = 0                               # ReLU 的导数
        dW1 = X.T @ dh
        W1 -= lr * dW1
        W2 -= lr * dW2
    h = np.maximum(0, X @ W1)                        # 推理：不加 mask
    return float(np.mean((h @ W2 - y) ** 2)), float(np.mean(np.abs(W1)))


for p in [0.0, 0.2, 0.5]:
    mse_tr, wmag = train_mlp(dropout_p=p)
    print(f"  dropout p={p:<4} 训练 MSE={mse_tr:.4f}  |W1| 均值={wmag:.4f}")

print("\n注意：dropout 会让训练误差变大（这是正常的），换来的是泛化更好。")

section("4) Early Stopping：盯住验证集，见好就收")
r = np.random.default_rng(9)
Xd = r.normal(size=(60, 30))
yd = Xd @ r.normal(size=(30, 1)) + r.normal(0, 0.3, (60, 1))
Xtr, Xva = Xd[:40], Xd[40:]
ytr2, yva2 = yd[:40], yd[40:]

W = r.normal(0, 0.15, (30, 1))
best_va, best_step, best_W, patience, bad = np.inf, -1, None, 20, 0
for step in range(2000):
    g = 2.0 * Xtr.T @ (Xtr @ W - ytr2) / len(Xtr)
    W -= 0.05 * g
    va = float(np.mean((Xva @ W - yva2) ** 2))
    if va < best_va - 1e-8:
        best_va, best_step, best_W, bad = va, step, W.copy(), 0
    else:
        bad += 1
        if bad >= patience:
            break
tr_final = float(np.mean((Xtr @ W - ytr2) ** 2))
tr_best = float(np.mean((Xtr @ best_W - ytr2) ** 2))
print(f"  一直训到底: step=1999  训练 MSE={tr_final:.4f}  验证 MSE={float(np.mean((Xva @ W - yva2) ** 2)):.4f}")
print(f"  早停:       step={best_step:<5} 训练 MSE={tr_best:.4f}  验证 MSE={best_va:.4f}  ← 更优")

section("5) 结论速记")
print("""
  · 训练误差下降≠模型变好，必须有一份"模型没见过"的数据
  · L2/权重衰减 = 让权重变小 → 函数更平滑 → 泛化更好
  · Dropout 在 Transformer 里写在每个子层输出上（p≈0.1），推理时自动关闭
  · Early Stopping 最便宜；LLM 因为训练一次太贵，通常改为"每隔 N 步存 checkpoint 再挑"
""")
