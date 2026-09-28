"""
05 · 卷积核的"物理含义"与特征图可视化
=======================================
卷积核不是抽象数字，每一个都有明确的图像处理含义。
理解它们，你就能读懂 CNN 第一层到底在学习什么。

本脚本手工实现 8 个经典核，在一张合成图上跑一遍，用 ASCII 热力图直接看效果
（不需要图形界面，SSH / Docker 里也能看）。
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import ascii_heatmap, section, set_seed, subsection

set_seed(8)

SIZE = 20


def make_image(size=SIZE):
    """合成一张有：亮方块 + 对角边缘 + 噪点的图。"""
    img = np.zeros((size, size))
    img[3:9, 3:9] = 1.0                      # 左上角亮方块
    for i in range(size):                    # 对角线
        if 0 <= i < size:
            img[i, i] = 0.8
    img[12:18, 12:18] = 0.5                  # 右下角灰方块
    return img + np.random.default_rng(0).normal(0, 0.02, (size, size))


img = make_image()


def conv2d_single(x, k, pad=1):
    """单通道 2D 卷积，same padding。"""
    H, W = x.shape
    KH, KW = k.shape
    xp = np.pad(x, pad)
    out = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            out[i, j] = np.sum(xp[i:i + KH, j:j + KW] * k)
    return out


KERNELS = {
    "恒等 identity":       np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]),
    "盒式模糊 box blur":   np.ones((3, 3)) / 9,
    "高斯模糊 gaussian":   np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16,
    "Sobel 垂直边缘":      np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]),
    "Sobel 水平边缘":      np.array([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]),
    "Laplacian 二阶导":    np.array([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]),
    "锐化 sharpen":        np.array([[0., -1., 0.], [-1., 5., -1.], [0., -1., 0.]]),
    "浮雕 emboss":         np.array([[-2., -1., 0.], [-1., 1., 1.], [0., 1., 2.]]),
}

section("1) 原始图像")
print(ascii_heatmap(img, width=44, height=18))

section("2) 每个核的输出（灰度越亮=响应越强）")
for name, k in KERNELS.items():
    out = conv2d_single(img, k)
    subsection(f"■ {name}")
    print(ascii_heatmap(out, width=44, height=18))
    print(f"  响应范围: [{out.min():.2f}, {out.max():.2f}]  核和={k.sum():.2f}")

section("3) 通道维度：RGB 图像上的卷积")
rgb = np.stack([img, np.roll(img, 3, axis=1), 1 - img])       # 假造 3 通道
print(f"  RGB 输入形状 = {rgb.shape}  (C,H,W)")
w3 = np.random.default_rng(1).normal(0, 0.3, (3, 3, 3))       # (C,KH,KW)：一个 3 通道核
out1 = sum(conv2d_single(rgb[c], w3[c]) for c in range(3))
print(f"  一个 3 通道卷积核 → 输出形状 = {out1.shape}  （多通道求和 = 1 张特征图）")
print("  关键点：卷积核的通道数必须等于输入通道数；输出通道数 = 卷积核的个数。")

section("4) 特征图堆叠与感受野")
feat = [conv2d_single(img, k) for k in [KERNELS["Sobel 垂直边缘"], KERNELS["Sobel 水平边缘"],
                                        KERNELS["盒式模糊 box blur"], KERNELS["Laplacian 二阶导"]]]
stack = np.stack(feat)
print(f"  4 个核 → 特征图形状 = {stack.shape}  (C=4, H, W)")
print(ascii_heatmap(stack.transpose(1, 0, 2).reshape(SIZE, 4 * SIZE), width=44, height=10,
                    title="  把 4 张特征图排成 2×2 看（可以看到不同核各司其职）："))

subsection("5) 两层卷积的感受野计算")
print("""
  感受野递推公式：   RF_{l} = RF_{l-1} + (K_l − 1) × Π_{i<l} stride_i

  例：3×3,stride=1 → 3×3,stride=1 → 3×3,stride=1
      第1层 RF=3；第2层 RF=3+(3−1)=5；第3层 RF=5+(3−1)=7
  这就是 "三个 3×3 顶一个 7×7" 的由来：参数 27 vs 49，非线性也更多。
""")

section("6) 结论速记")
print("""
  · 浅层卷积核 ≈ 边缘/纹理检测器（Gabor 滤波器）；中层 ≈ 部件；深层 ≈ 语义
  · 训练好的 CNN 第一层真的会长出 Sobel 样的核 —— 下一节我们用代码验证这件事
  · 可视化特征图是调试 CNN 最有效的手段之一（比看 loss 曲线有用得多）
""")
