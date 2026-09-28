"""
03 · 逻辑回归：从"回归"到"分类"，也是理解交叉熵的最短路径
=========================================================
模型：z = Xw + b ; p = σ(z) = 1/(1+e⁻ᶻ)
损失：BCE = −mean( y·log p + (1−y)·log(1−p) )
梯度：dL/dz = (p − y)/n          ← 这个式子极其简洁，因为 σ' 被抵消了

本脚本：
  1. 手写前向、损失、梯度，并用有限差分做梯度检查
  2. 训练二分类器，打印准确率和混淆矩阵
  3. 展示"交叉熵 vs 均方误差"在分类问题上谁更好
  4. 画出决策边界（有 matplotlib 时）
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import check_close, grad_check, has_matplotlib, save_fig, section, set_seed, subsection

set_seed(3)

# ----------------------------------------------------------------------------------
# 两类高斯团
# ----------------------------------------------------------------------------------
N_PER = 150
mu1, mu2 = np.array([-1.6, -1.6]), np.array([1.6, 1.6])
X1 = np.random.randn(N_PER, 2) + mu1
X2 = np.random.randn(N_PER, 2) + mu2
X = np.vstack([X1, X2])
y = np.hstack([np.zeros(N_PER), np.ones(N_PER)])

# 标准化
mu, sigma = X.mean(0), X.std(0)
Xn = (X - mu) / sigma
Xb = np.hstack([Xn, np.ones((X.shape[0], 1))])        # 偏置吸收进 w
n, d = Xb.shape


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def forward(w: np.ndarray) -> np.ndarray:
    return sigmoid(Xb @ w)


def bce(w: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(forward(w), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def grad_bce(w: np.ndarray) -> np.ndarray:
    p = forward(w)
    return Xb.T @ (p - y) / n


section("1) 梯度检查：手写的 dL/dw 对不对？用有限差分验证")
w0 = np.random.randn(d) * 0.1
analytic = grad_bce(w0)
grad_check(lambda w: bce(w), w0, analytic, eps=1e-6, tol=1e-5, name="dL/dw (逻辑回归)")
print("  提示：以后写任何反向传播，第一步永远是先做 grad check。")

section("2) 训练：全批量梯度下降")
w = np.zeros(d)
lr, steps = 1.0, 800
for step in range(steps):
    w -= lr * grad_bce(w)
    if step % 200 == 0 or step == steps - 1:
        p = forward(w)
        acc = float(np.mean((p >= 0.5).astype(float) == y))
        print(f"  step {step:>4}  loss={bce(w):.5f}  acc={acc:.4f}")

p = forward(w)
pred = (p >= 0.5).astype(float)
tp = int(np.sum((pred == 1) & (y == 1)))
tn = int(np.sum((pred == 0) & (y == 0)))
fp = int(np.sum((pred == 1) & (y == 0)))
fn = int(np.sum((pred == 0) & (y == 1)))
print(f"\n混淆矩阵: TP={tp}  FP={fp}  FN={fn}  TN={tn}")
print(f"准确率={(tp + tn) / n:.4f}  精确率={tp / (tp + fp):.4f}  召回率={tp / (tp + fn):.4f}")

subsection("3) 决策边界：w₀x₁ + w₁x₂ + b = 0  →  x₂ = −(w₀x₁ + b)/w₁")
print(f"  学到的 w={np.round(w[:2], 4)}, b={w[2]:.4f}")
print(f"  边界斜率={-w[0] / w[1]:.4f}, 截距={-w[2] / w[1]:.4f}")

section("4) 分类为什么用交叉熵而不是 MSE？各训 800 步对比")
w_mse = np.zeros(d)
for step in range(steps):
    p_mse = forward(w_mse)                       # 仍然用 sigmoid 输出概率
    dL_dz = 2.0 * (p_mse - y) / n * (p_mse * (1 - p_mse))   # MSE 的梯度带 σ' 因子
    w_mse -= 1.0 * (Xb.T @ dL_dz)
print(f"  交叉熵 loss={bce(w):.5f}   acc={np.mean((forward(w) >= .5) == y):.4f}")
print(f"  MSE    loss={bce(w_mse):.5f}   acc={np.mean((forward(w_mse) >= .5) == y):.4f}")
print("  原因：MSE 的梯度含 σ'(z)，当预测极端错误时 σ'≈0 → 梯度消失，学不动。")

if has_matplotlib():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.scatter(Xn[y == 0, 0], Xn[y == 0, 1], s=16, alpha=0.7, label="class 0")
    ax.scatter(Xn[y == 1, 0], Xn[y == 1, 1], s=16, alpha=0.7, label="class 1")
    xs = np.linspace(Xn[:, 0].min() - 0.5, Xn[:, 0].max() + 0.5, 100)
    ax.plot(xs, -(w[0] * xs + w[2]) / w[1], "r-", lw=2, label="决策边界")
    ax.legend()
    ax.set_title("逻辑回归决策边界")
    save_fig(fig, Path(__file__).parent / "outputs" / "03_logistic_boundary.png")

section("5) 结论速记")
print("""
  · 逻辑回归 = 线性层 + sigmoid + 交叉熵；它是神经网络单个神经元的完整形态
  · sigmoid + BCE 的梯度恰好是 (p − y)，这个"简洁"是刻意选择配对损失函数的结果
  · 后面 LLM 的输出层做的正是"逻辑回归的 5 万分类版本"（vocab 上的 softmax + 交叉熵）
""")
