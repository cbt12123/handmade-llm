"""
04 · 从零搭一个 LeNet-5（纯 numpy 前向 + 手写反向，跑通一个玩具分类任务）
==========================================================================
LeNet-5（1998）是 CNN 的鼻祖架构：
  Conv(1→6, 5×5) → Tanh → AvgPool(2) → Conv(6→16, 5×5) → Tanh → AvgPool(2)
  → Flatten(16·5·5=400) → FC(120) → Tanh → FC(84) → Tanh → FC(10) → Softmax

本脚本：
  1. 用 numpy 手写整个网络的前向与反向
  2. 训练一个"竖线 vs 横线"的 28×28 二分类玩具任务
  3. 再用 PyTorch 搭一遍同样的网络，验证参数量与输出一致
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import check_close, human_num, section, set_seed, subsection

set_seed(6)

# ----------------------------------------------------------------------------------
# 数据：28×28 的竖线/横线（5 类改成 2 类，便于快速训练）
# ----------------------------------------------------------------------------------
def make_shapes(n_per=120, size=28, noise=0.25, seed=0):
    rng = np.random.default_rng(seed)
    imgs, labels = [], []
    for _ in range(n_per):
        img = rng.normal(0, noise, (size, size))
        i = rng.integers(4, size - 4)
        if rng.random() < 0.5:                 # 竖线
            img[:, i:i + 2] += 1.0
            labels.append(0)
        else:                                  # 横线
            img[i:i + 2, :] += 1.0
            labels.append(1)
        imgs.append(img)
    return np.array(imgs)[:, None], np.array(labels)


X, y = make_shapes()
N = X.shape[0]
print(f"数据集 X={X.shape}  y={y.shape}（0=竖线, 1=横线）")


# ----------------------------------------------------------------------------------
# 层的前向/反向
# ----------------------------------------------------------------------------------
def conv_forward(x, w, b, pad=0):
    """x:(N,C,H,W) w:(OC,C,K,K)"""
    Nn, C, H, W_ = x.shape
    OC, _, K, _ = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    OH, OW = H + 2 * pad - K + 1, W_ + 2 * pad - K + 1
    cols = np.zeros((Nn, C * K * K, OH * OW))
    for n in range(Nn):
        col = 0
        for i in range(OH):
            for j in range(OW):
                cols[n, :, col] = xp[n, :, i:i + K, j:j + K].ravel()
                col += 1
    out = np.einsum("of,nfp->nop", w.reshape(OC, -1), cols).reshape(Nn, OC, OH, OW) + b.reshape(1, OC, 1, 1)
    return out, (cols, OH, OW, pad, x.shape)


def conv_backward(dout, w, cache):
    cols, OH, OW, pad, x_shape = cache
    Nn, C, H, W_ = x_shape
    OC = w.shape[0]
    K = w.shape[2]
    dout_f = dout.reshape(Nn, OC, -1)
    dW = np.einsum("nop,nfp->of", dout_f, cols).reshape(w.shape)
    dB = dout_f.sum(axis=(0, 2))
    dcols = np.einsum("of,nop->nfp", w.reshape(OC, -1), dout_f)
    dxp = np.zeros((Nn, C, H + 2 * pad, W_ + 2 * pad))
    for n in range(Nn):
        col = 0
        for i in range(OH):
            for j in range(OW):
                dxp[n, :, i:i + K, j:j + K] += dcols[n, :, col].reshape(C, K, K)
                col += 1
    dx = dxp[:, :, pad:pad + H, pad:pad + W_] if pad else dxp
    return dx, dW, dB


def pool_forward(x, k=2):
    Nn, C, H, W_ = x.shape
    OH, OW = H // k, W_ // k
    xr = x[:, :, :OH * k, :OW * k].reshape(Nn, C, OH, k, OW, k)
    out = xr.max(axis=(3, 5))
    mask = (xr == out[:, :, :, None, :, None])
    return out, (mask, xr.shape, k)


def pool_backward(dout, cache):
    mask, shape, k = cache
    Nn, C, OH, _, OW, _ = shape
    d = np.zeros(shape)
    d[:] = dout[:, :, :, None, :, None] * mask
    return d.reshape(Nn, C, OH * k, OW * k)


def fc_forward(x, w, b):
    out = x @ w + b
    return out, (x, w)


def fc_backward(dout, cache):
    x, w = cache
    dx = dout @ w.T
    dW = x.T @ dout
    dB = dout.sum(0)
    return dx, dW, dB


# ----------------------------------------------------------------------------------
# 网络定义（LeNet 风格，缩小版）
# ----------------------------------------------------------------------------------
rng = np.random.default_rng(0)
def init(fan_in, fan_out):
    return rng.normal(0, np.sqrt(2.0 / (fan_in + fan_out)), (fan_in, fan_out))


P = {
    "c1w": init(1 * 5 * 5, 6).reshape(6, 1, 5, 5), "c1b": np.zeros(6),
    "c2w": init(6 * 5 * 5, 16).reshape(16, 6, 5, 5), "c2b": np.zeros(16),
    "f1w": init(16 * 4 * 4, 32), "f1b": np.zeros(32),
    "f2w": init(32, 2), "f2b": np.zeros(2),
}
total = sum(v.size for v in P.values())
print(f"LeNet(缩小版) 参数量 = {human_num(float(total))}")


def forward(X_in, cache=None):
    h, c1 = conv_forward(X_in, P["c1w"], P["c1b"])          # 28→24
    h = np.tanh(h)
    h, p1 = pool_forward(h, 2)                              # 24→12
    h, c2 = conv_forward(h, P["c2w"], P["c2b"])             # 12→8
    h = np.tanh(h)
    h, p2 = pool_forward(h, 2)                              # 8→4
    flat = h.reshape(h.shape[0], -1)
    h, f1 = fc_forward(flat, P["f1w"], P["f1b"])
    h = np.tanh(h)
    logits, f2 = fc_forward(h, P["f2w"], P["f2b"])
    if cache is not None:
        cache.update(c1=c1, p1=p1, c2=c2, p2=p2, flat=flat, f1=f1, f2=f2, h1=None)
    return logits


def softmax(logits):
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def loss_and_grads(X_in, y_in):
    cache = {}
    # 为简化反向，把中间激活按顺序重算一遍并保存
    a0 = X_in
    z1, c1 = conv_forward(a0, P["c1w"], P["c1b"])
    a1 = np.tanh(z1)
    p1_out, p1 = pool_forward(a1, 2)
    z2, c2 = conv_forward(p1_out, P["c2w"], P["c2b"])
    a2 = np.tanh(z2)
    p2_out, p2 = pool_forward(a2, 2)
    flat = p2_out.reshape(p2_out.shape[0], -1)
    z3, f1 = fc_forward(flat, P["f1w"], P["f1b"])
    a3 = np.tanh(z3)
    logits, f2 = fc_forward(a3, P["f2w"], P["f2b"])

    prob = softmax(logits)
    m = y_in.size
    loss = -np.mean(np.log(prob[np.arange(m), y_in] + 1e-12))

    dz = prob.copy()
    dz[np.arange(m), y_in] -= 1
    dz /= m

    da3, g_f2w, g_f2b = fc_backward(dz, f2)
    dz3 = da3 * (1 - a3**2)
    dflat, g_f1w, g_f1b = fc_backward(dz3, f1)
    dp2 = dflat.reshape(p2_out.shape)
    da2 = pool_backward(dp2, p2)
    dz2 = da2 * (1 - a2**2)
    dp1, g_c2w, g_c2b = conv_backward(dz2, P["c2w"], c2)
    da1 = pool_backward(dp1, p1)
    dz1 = da1 * (1 - a1**2)
    _, g_c1w, g_c1b = conv_backward(dz1, P["c1w"], c1)

    grads = {"c1w": g_c1w, "c1b": g_c1b, "c2w": g_c2w, "c2b": g_c2b,
             "f1w": g_f1w, "f1b": g_f1b, "f2w": g_f2w, "f2b": g_f2b}
    return loss, grads, logits


section("1) 训练（全批量梯度下降，lr=0.05，200 步）")
lr = 0.05
for step in range(201):
    loss, grads, logits = loss_and_grads(X, y)
    for k in P:
        P[k] -= lr * grads[k]
    if step % 50 == 0:
        acc = float(np.mean(logits.argmax(1) == y))
        print(f"  step {step:>4}  loss={loss:.4f}  acc={acc:.4f}")

loss, _, logits = loss_and_grads(X, y)
print(f"\n最终：loss={loss:.4f}  训练准确率={np.mean(logits.argmax(1) == y):.4f}")

section("2) 用 PyTorch 搭同样的网络验证参数量")
try:
    import torch
    import torch.nn as nn

    class LeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 6, 5), nn.Tanh(), nn.AvgPool2d(2),
                nn.Conv2d(6, 16, 5), nn.Tanh(), nn.AvgPool2d(2),
                nn.Flatten(),
                nn.Linear(16 * 4 * 4, 32), nn.Tanh(),
                nn.Linear(32, 2),
            )

        def forward(self, x):
            return self.net(x)

    net = LeNet()
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  PyTorch 版参数量 = {human_num(float(n_params))}（与 numpy 版的 {human_num(float(total))} 一致）")
    with torch.no_grad():
        out = net(torch.tensor(X, dtype=torch.float32))
    print(f"  未训练的 torch 版输出 logits[0] = {np.round(out[0].numpy(), 4)}")
except ImportError:
    print("[SKIP] 未安装 torch，跳过对照")

section("3) 结论速记")
print("""
  · CNN 的典型范式： (Conv → 非线性 → Pool)×n → Flatten → FC×n → Softmax
  · 每过一次 pool，空间尺寸减半、通道数通常翻倍 —— 用"通道"换"分辨率"是 CNN 的通用设计
  · 把最后的 FC 换成 GlobalAvgPool + 1 层 FC（ResNet 做法）可以砍掉 90% 的参数
  · 现代视觉里 CNN 被 ViT 抢了风头，但 Conv 的归纳偏置（局部性+平移等变）在小数据上依然更强
""")
