"""
01 · 卷积到底在算什么：手写 conv2d（朴素版 + im2col 版）
=========================================================
卷积 = "拿一个小窗口（卷积核）在图像上滑动，每次做一次点乘求和"。

三种实现方式，从慢到快：
  ① 四层 for 循环的朴素版          —— 最直观，用来建立直觉
  ② im2col + 一次大矩阵乘法（GEMM）—— cuDNN/CUDA 卷积的真实做法（第 7 章会写 matmul 算子）
  ③ torch.nn.functional.conv2d     —— 生产实现

顺便回答两个高频问题：输出尺寸怎么算？参数量怎么算？
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import check_close, human_num, section, set_seed, subsection, timer

set_seed(1)


def conv2d_naive(x, w, b=None, stride=1, pad=0):
    """朴素实现：x:(C,H,W)  w:(OC,C,KH,KW)  →  out:(OC,OH,OW)"""
    C, H, W = x.shape
    OC, _, KH, KW = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    OH = (H + 2 * pad - KH) // stride + 1
    OW = (W + 2 * pad - KW) // stride + 1
    out = np.zeros((OC, OH, OW))
    for oc in range(OC):                       # 每个卷积核产生一张特征图
        for oh in range(OH):
            for ow in range(OW):
                patch = xp[:, oh * stride:oh * stride + KH, ow * stride:ow * stride + KW]
                out[oc, oh, ow] = np.sum(patch * w[oc]) + (0 if b is None else b[oc])
    return out


def im2col(x, KH, KW, stride=1, pad=0):
    """把"滑动窗口取 patch"变成矩阵的一列：卷积 → 矩阵乘法 的关键一步。

    x:(N,C,H,W) → cols:(N, C·KH·KW, OH·OW)
    """
    N, C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    OH = (H + 2 * pad - KH) // stride + 1
    OW = (W + 2 * pad - KW) // stride + 1
    cols = np.zeros((N, C * KH * KW, OH * OW))
    for n in range(N):
        col = 0
        for i in range(OH):
            for j in range(OW):
                patch = xp[n, :, i * stride:i * stride + KH, j * stride:j * stride + KW]
                cols[n, :, col] = patch.ravel()
                col += 1
    return cols, OH, OW


def conv2d_im2col(x, w, b=None, stride=1, pad=0):
    """x:(N,C,H,W)  w:(OC,C,KH,KW) → out:(N,OC,OH,OW)"""
    N, C, H, W = x.shape
    OC, _, KH, KW = w.shape
    cols, OH, OW = im2col(x, KH, KW, stride, pad)
    w2d = w.reshape(OC, -1)                                    # (OC, C·KH·KW)
    out = np.einsum("of,nfp->nop", w2d, cols).reshape(N, OC, OH, OW)   # 一次 GEMM 完成所有窗口
    if b is not None:
        out += b.reshape(1, OC, 1, 1)
    return out


# ----------------------------------------------------------------------------------
section("1) 最小可运行的例子：3×3 图像 + 一个边缘检测核")
img = np.array([[1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0]])
kernel = np.array([[1.0, 0.0, -1.0],
                   [1.0, 0.0, -1.0],
                   [1.0, 0.0, -1.0]])
out = conv2d_naive(img[None], kernel[None, None])
print("  输入图像：")
print("\n".join("    " + " ".join(f"{v:5.1f}" for v in row) for row in img))
print("  卷积核（垂直边缘检测）：")
print("\n".join("    " + " ".join(f"{v:5.1f}" for v in row) for row in kernel))
print(f"  输出（1×1）: {out.ravel()}   ← 中间那条 1→0 的竖边被检测到了")

section("2) 多通道、多核、padding、stride")
N, C, H, W = 2, 3, 8, 8
OC, K = 4, 3
rng = np.random.default_rng(0)
X = rng.normal(size=(N, C, H, W)).astype(np.float32)
Wk = rng.normal(0, 0.1, (OC, C, K, K)).astype(np.float32)
bk = rng.normal(0, 0.1, OC).astype(np.float32)

o_naive = np.stack([conv2d_naive(X[n], Wk, bk) for n in range(N)])
o_im2col = conv2d_im2col(X, Wk, bk)
check_close(o_naive, o_im2col, tol=1e-5, name="朴素 conv vs im2col conv")

subsection("3) 输出尺寸公式（必须背）")
print("  OH = (H + 2·pad − dilation·(KH−1) − 1) / stride + 1")
for pad, stride in [(0, 1), (1, 1), (0, 2), (2, 1)]:
    OH = (H + 2 * pad - K) // stride + 1
    print(f"  H=8, K=3, pad={pad}, stride={stride} → OH={OH}")
print("  常见配置：K=3,pad=1,stride=1 → 尺寸不变（same）；K=3,pad=0,stride=2 → 尺寸减半")

subsection("4) 参数量与计算量")
params = OC * C * K * K + OC
print(f"  卷积层参数 = OC·C·KH·KW + OC = {OC}·{C}·{K}·{K}+{OC} = {params}")
print("  注意：卷积的参数量与输入图像大小无关 → 这就是『权值共享』，CNN 比全连接省参数的原因")
OH = OW = H - K + 1
macs = N * OC * OH * OW * C * K * K
print(f"  本例乘加数(MAC) = {human_num(macs)}，×2 即 FLOPs ≈ {human_num(2 * macs)}")
fc_params = H * W * C * (H * W * C)
print(f"  同等输入输出的全连接层参数会是 {human_num(float(fc_params))} 量级 —— 差了几个数量级")

subsection("5) 性能对比：朴素循环 vs im2col+GEMM")
Xb = rng.normal(size=(8, 16, 28, 28))
Wb = rng.normal(0, 0.1, (32, 16, 3, 3))
with timer("朴素四层循环"):
    _ = np.stack([conv2d_naive(Xb[n], Wb) for n in range(8)])
with timer("im2col + einsum"):
    _ = conv2d_im2col(Xb, Wb)
print("  真实框架里 im2col 这一步会被 CUDA kernel 直接融合掉（implicit GEMM），连展开都省了。")

section("6) 和 PyTorch 对照")
try:
    import torch
    import torch.nn.functional as F

    t_out = F.conv2d(torch.tensor(X), torch.tensor(Wk), torch.tensor(bk))
    check_close(o_im2col, t_out, tol=1e-5, name="手写 conv vs torch F.conv2d")
except ImportError:
    print("[SKIP] 未安装 torch，跳过对照")

section("7) 结论速记")
print("""
  · 卷积 = 局部连接 + 权值共享：每个输出像素只看输入的 K×K 邻域，且所有位置共用同一个核
  · im2col 把卷积变成矩阵乘法 → 于是"卷积加速"就变成了"GEMM 加速"（第 7 章的 tiled matmul）
  · 感受野：堆叠两个 3×3 卷积 = 一个 5×5 的感受野，但参数更少 → VGG 的核心思想
  · Transformer 崛起后，Conv 退居视觉/局部特征；但 Conv 的"局部性 + 权值共享"思想
    被重新搬进了 LLM：局部注意力、滑动窗口、深度可分离卷积（如 RWKV 的 token shift）
""")
