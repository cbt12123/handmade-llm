"""
03 · 卷积的反向传播（手写 dW / dX）
=====================================
卷积层有两个可导对象：
  dL/dW ：把每个 patch 与对应的输出梯度相乘再求和  → dW = Σ dout·patch
  dL/dX ：把输出梯度"散回"到输入位置（等价于用翻转的核做 full 卷积）

本脚本手写两者，并用有限差分严格验证 —— 这和第 1 章验证 MLP 是同一套方法。
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import check_close, grad_check, section, set_seed, subsection

set_seed(4)

N, C, H, W = 1, 2, 6, 6
OC, K = 3, 3


def conv_forward(x, w, b, pad=1):
    """x:(N,C,H,W) w:(OC,C,K,K) → out, cache"""
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    OH, OW = H + 2 * pad - K + 1, W + 2 * pad - K + 1
    cols = np.zeros((N, C * K * K, OH * OW))
    for n in range(N):
        col = 0
        for i in range(OH):
            for j in range(OW):
                cols[n, :, col] = xp[n, :, i:i + K, j:j + K].ravel()
                col += 1
    out = np.einsum("of,nfp->nop", w.reshape(OC, -1), cols).reshape(N, OC, OH, OW) + b.reshape(1, OC, 1, 1)
    return out, (cols, OH, OW, pad)


def conv_backward(dout, x_shape, w, cache):
    cols, OH, OW, pad = cache
    Nn = dout.shape[0]
    dout_f = dout.reshape(Nn, w.shape[0], -1)             # (N, OC, OH·OW)
    # ---- dW: (OC, C·K·K) = dout(展平) @ cols^T ----
    dW = np.einsum("nop,nfp->of", dout_f, cols).reshape(w.shape)
    dB = dout_f.sum(axis=(0, 2))
    # ---- dX: 把 dout 乘回权重，再 scatter 到输入位置 ----
    dcols = np.einsum("of,nop->nfp", w.reshape(OC, -1), dout_f)
    dxp = np.zeros((Nn, C, H + 2 * pad, W + 2 * pad))
    for n in range(Nn):
        col = 0
        for i in range(OH):
            for j in range(OW):
                dxp[n, :, i:i + K, j:j + K] += dcols[n, :, col].reshape(C, K, K)
                col += 1
    dx = dxp[:, :, pad:pad + H, pad:pad + W] if pad else dxp
    return dx, dW, dB


rng = np.random.default_rng(0)
X = rng.normal(size=(N, C, H, W))
Wk = rng.normal(0, 0.2, (OC, C, K, K))
B = rng.normal(0, 0.1, OC)
dout = rng.normal(size=(N, OC, H, W))          # 假装这是上一层传回来的梯度

section("1) 前向 + 反向")
out, cache = conv_forward(X, Wk, B)
dx, dW, dB = conv_backward(dout, X.shape, Wk, cache)
print(f"  out 形状 {out.shape}   dx 形状 {dx.shape}   dW 形状 {dW.shape}   dB 形状 {dB.shape}")
print("  形状必须和输入/参数一一对应，这是检查反向传播是否写错的第一步。")

section("2) 有限差分验证 dW")
flat_W = Wk.ravel()


def loss_on_W(flat):
    return float(np.sum(conv_forward(X, flat.reshape(Wk.shape), B)[0] * dout))


grad_check(loss_on_W, flat_W, dW.ravel(), eps=1e-6, tol=1e-6, name="dW（卷积核梯度）")

section("3) 有限差分验证 dX")
flat_X = X.ravel()


def loss_on_X(flat):
    return float(np.sum(conv_forward(flat.reshape(X.shape), Wk, B)[0] * dout))


grad_check(loss_on_X, flat_X, dx.ravel(), eps=1e-6, tol=1e-6, name="dX（输入梯度）")

section("4) 和 PyTorch 的 autograd 对照")
try:
    import torch
    import torch.nn.functional as F

    Xt = torch.tensor(X, requires_grad=True)
    Wt = torch.tensor(Wk, requires_grad=True)
    Bt = torch.tensor(B, requires_grad=True)
    ot = F.conv2d(Xt, Wt, Bt, padding=1)
    ot.backward(torch.tensor(dout))
    check_close(dW, Wt.grad, tol=1e-6, name="dW 手写 vs torch")
    check_close(dx, Xt.grad, tol=1e-6, name="dX 手写 vs torch")
    check_close(dB, Bt.grad, tol=1e-6, name="dB 手写 vs torch")
except ImportError:
    print("[SKIP] 未安装 torch，跳过对照")

section("5) 直觉解释")
print("""
  · dW = Σ(输出梯度 × 对应 patch)  ：某个卷积核权重，对所有"它扫过的位置"的误差负责
  · dX = 用翻转 180° 的核卷积 dout ：误差沿着卷积的路径原路传回输入（full 卷积）
  · 一旦写成 im2col 形式，dW 和 dX 都退化成矩阵乘法 → GPU 上就能用现成的 GEMM 算子
  · 真实训练里还会做"梯度累加"（grad accumulation），把大 batch 拆成多次前向再求和
""")
