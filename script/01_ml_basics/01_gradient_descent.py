"""
01 · 梯度下降：所有深度学习唯一的"发动机"
=========================================
你只需要懂 Python 循环就能读懂这个脚本：梯度下降的本质就是

        w <- w - lr * dL/dw          （沿着"下坡方向"挪一小步，循环 N 次）

本脚本做三件事：
  1. 在一个二维凸函数上，手推 + 手写 4 个优化器：GD / Momentum / RMSProp / Adam
  2. 观察学习率过大 → 发散、过小 → 收敛慢
  3. 给出"什么时候用哪个优化器"的直觉
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import has_matplotlib, save_fig, section, set_seed, subsection

set_seed(0)

# ----------------------------------------------------------------------------------
# 目标函数：f(w) = 1/2 wᵀAw − bᵀw，A 正定 → 碗形，最小值 w* = A⁻¹b（可解析求出）
# ----------------------------------------------------------------------------------
A = np.array([[3.0, 1.0], [1.0, 2.0]])
B = np.array([[1.0], [-2.0]])


def loss(w: np.ndarray) -> float:
    return float((0.5 * w.T @ A @ w - B.T @ w).item())   # w 是 (2,1)，结果是 (1,1)，取标量


def grad(w: np.ndarray) -> np.ndarray:
    return A @ w - B


W_STAR = np.linalg.solve(A, B)
print(f"解析最优解 w* = {W_STAR.ravel()},  f(w*) = {loss(W_STAR):.6f}")


# ----------------------------------------------------------------------------------
# 四个优化器：全部只用 numpy 的加减乘除，没有任何"魔法"
# ----------------------------------------------------------------------------------
def sgd(w0, lr, steps, beta_momentum=0.0):
    """带动量的 SGD：v = μ·v − lr·g ; w = w + v
    μ=0 时退化成最朴素的梯度下降。"""
    w, v = w0.copy(), np.zeros_like(w0)
    hist = [loss(w)]
    for _ in range(steps):
        g = grad(w)
        v = beta_momentum * v - lr * g
        w = w + v
        hist.append(loss(w))
    return w, np.array(hist)


def rmsprop(w0, lr, steps, beta=0.99, eps=1e-8):
    """RMSProp：用梯度的二阶动量把每个维度"归一化"，适合各维度曲率差异大的地形。"""
    w = w0.copy()
    s = np.zeros_like(w0)
    hist = [loss(w)]
    for _ in range(steps):
        g = grad(w)
        s = beta * s + (1 - beta) * (g * g)          # 梯度平方的滑动平均
        w = w - lr * g / (np.sqrt(s) + eps)          # 曲率大的方向步子变小
        hist.append(loss(w))
    return w, np.array(hist)


def adam(w0, lr, steps, b1=0.9, b2=0.999, eps=1e-8):
    """Adam = Momentum(一阶动量) + RMSProp(二阶动量) + 偏差修正。LLM 训练的默认选择。"""
    w = w0.copy()
    m = np.zeros_like(w0)
    v = np.zeros_like(w0)
    hist = [loss(w)]
    for t in range(1, steps + 1):
        g = grad(w)
        m = b1 * m + (1 - b1) * g                    # 一阶动量（方向惯性）
        v = b2 * v + (1 - b2) * (g * g)              # 二阶动量（逐维度缩放）
        m_hat = m / (1 - b1**t)                      # 偏差修正：前几步 m,v 偏小
        v_hat = v / (1 - b2**t)
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
        hist.append(loss(w))
    return w, np.array(hist)


section("1) 固定学习率 lr=0.1，比较 4 个优化器（200 步，起点 (-3, 3)）")
W0 = np.array([[-3.0], [3.0]])
results = {
    "GD":        sgd(W0, lr=0.1, steps=200, beta_momentum=0.0),
    "Momentum":  sgd(W0, lr=0.1, steps=200, beta_momentum=0.9),
    "RMSProp":   rmsprop(W0, lr=0.1, steps=200),
    "Adam":      adam(W0, lr=0.1, steps=200),
}
print(f"{'优化器':<10}{'最终 w':<28}{'最终 loss':>14}{'到 w* 距离':>14}")
for name, (w, hist) in results.items():
    print(f"{name:<10}{str(np.round(w.ravel(), 5)):<28}{hist[-1]:>14.3e}"
          f"{np.linalg.norm(w - W_STAR):>14.3e}")

subsection("2) 学习率扫描：同一个优化器，lr 决定生死")
print(f"{'lr':<8}{'GD(200步后 loss)':>22}{'判定':>12}")
for lr in [0.01, 0.05, 0.1, 0.3, 0.5, 0.8]:
    w, hist = sgd(W0, lr=lr, steps=200, beta_momentum=0.0)
    verdict = "发散/震荡" if not np.isfinite(hist[-1]) or hist[-1] > 1e3 else "收敛"
    print(f"{lr:<8}{hist[-1]:>22.4e}{verdict:>12}")

subsection("3) 为什么需要 Momentum：看前 20 步的『之字形』路径")
_, hist_gd = sgd(W0, lr=0.1, steps=20, beta_momentum=0.0)
_, hist_mom = sgd(W0, lr=0.1, steps=20, beta_momentum=0.9)
print("  step   GD loss      Momentum loss")
for i in range(0, 21, 4):
    print(f"  {i:<5}{hist_gd[i]:>12.5f}{hist_mom[i]:>16.5f}")

# ----------------------------------------------------------------------------------
# 可选：把优化轨迹画在等高线上
# ----------------------------------------------------------------------------------
if has_matplotlib():
    import matplotlib.pyplot as plt

    xs = np.linspace(-4, 4, 200)
    ys = np.linspace(-4, 4, 200)
    X, Y = np.meshgrid(xs, ys)
    Z = 0.5 * (A[0, 0] * X**2 + (A[0, 1] + A[1, 0]) * X * Y + A[1, 1] * Y**2) - (B[0, 0] * X + B[1, 0] * Y)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (name, (w, _)) in zip(axes, [("GD", results["GD"]), ("Adam", results["Adam"])]):
        ax.contour(X, Y, Z, levels=25, cmap="viridis", alpha=0.6)
        traj = [W0.ravel()]
        ww = W0.copy()
        # 重新跑一遍以记录轨迹（与上面同样公式，保持教学透明）
        m = np.zeros_like(ww)
        v = np.zeros_like(ww)
        for t in range(1, 61):
            g = grad(ww)
            if name == "GD":
                ww = ww - 0.1 * g
            else:
                m = 0.9 * m + 0.1 * g
                v = 0.999 * v + 0.001 * (g * g)
                ww = ww - 0.1 * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
            traj.append(ww.ravel())
        traj = np.array(traj)
        ax.plot(traj[:, 0], traj[:, 1], "ro-", markersize=2.5, linewidth=1)
        ax.plot(*W_STAR.ravel(), "b*", markersize=14, label="w* (最优解)")
        ax.set_title(f"{name} 优化轨迹")
        ax.legend()
        ax.set_xlabel("w1")
        ax.set_ylabel("w2")
    save_fig(fig, Path(__file__).parent / "outputs" / "01_optimizer_paths.png")
else:
    print("[info] 未安装 matplotlib，跳过绘图（不影响其余结论）")

section("4) 结论速记")
print("""
  · 梯度下降 = 反复执行 `w -= lr * grad`，深度学习和它唯一的区别是 grad 用链式法则算出来的
  · 学习率 lr 是最重要的超参数：太大发散、太小收敛慢；大模型训练还要配合 warmup + 余弦衰减
  · Momentum 解决"峡谷地形来回震荡"，RMSProp 解决"各维度曲率不一致"
  · Adam = 两者相加 + 偏差修正；Transformer/LLM 几乎默认 AdamW（Adam + 解耦权重衰减）
""")
