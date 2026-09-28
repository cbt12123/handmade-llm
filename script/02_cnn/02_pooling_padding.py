"""
02 · Padding / Stride / Pooling / 转置卷积：CNN 的"空间操作工具箱"
=====================================================================
这些操作本身没有参数（除了转置卷积），但决定了特征图的尺寸和信息流动：
  padding   : 补边，防止越卷越小、也保护边缘信息
  stride    : 步长 >1 直接下采样（等价于"跳过"一些位置）
  pooling   : 聚合降采样，带来平移不变性 + 省算力
  dilation  : 空洞卷积，不增加参数就扩大感受野
  transposed conv : 上采样，用于分割 / 生成器
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import ascii_heatmap, section, set_seed, subsection

set_seed(2)


def max_pool2d(x, k=2, stride=None):
    """x:(C,H,W) → (C,H/k,W/k)。返回池化结果（这里不返回索引，简化版）。"""
    stride = stride or k
    C, H, W = x.shape
    OH, OW = (H - k) // stride + 1, (W - k) // stride + 1
    out = np.zeros((C, OH, OW))
    for c in range(C):
        for i in range(OH):
            for j in range(OW):
                out[c, i, j] = np.max(x[c, i * stride:i * stride + k, j * stride:j * stride + k])
    return out


def avg_pool2d(x, k=2, stride=None):
    stride = stride or k
    C, H, W = x.shape
    OH, OW = (H - k) // stride + 1, (W - k) // stride + 1
    out = np.zeros((C, OH, OW))
    for c in range(C):
        for i in range(OH):
            for j in range(OW):
                out[c, i, j] = np.mean(x[c, i * stride:i * stride + k, j * stride:j * stride + k])
    return out


def global_avg_pool(x):
    """ GAP: 每张特征图取一个均值 → (C,)。分类网络用它替代全连接，参数量骤降。"""
    return x.reshape(x.shape[0], -1).mean(axis=1)


# ----------------------------------------------------------------------------------
section("1) 手工构造一张 8×8 图像（中间一个亮方块），观察各种操作")
img = np.zeros((1, 8, 8))
img[0, 2:6, 2:6] = 1.0
print(ascii_heatmap(img[0], width=32, height=12, title="原图 8×8："))

print(ascii_heatmap(max_pool2d(img, 2)[0], width=32, height=12, title="\nMaxPool 2×2（4×4）："))
print(ascii_heatmap(avg_pool2d(img, 2)[0], width=32, height=12, title="\nAvgPool 2×2（4×4）："))
print("  MaxPool 保留『最亮的那个像素』（锐利，利于纹理），AvgPool 做平滑（利于整体形状）。")

# ----------------------------------------------------------------------------------
section("2) Padding 的三种用法")
def pad2d(x, pad, mode="constant", value=0.0):
    kwargs = {"constant_values": value} if mode == "constant" else {}
    return np.pad(x, ((0, 0), (pad, pad), (pad, pad)), mode=mode, **kwargs)


x = np.arange(16, dtype=float).reshape(1, 4, 4)
print("  原图 4×4：")
print("\n".join("    " + " ".join(f"{v:4.0f}" for v in row) for row in x[0]))
print("\n  pad=1, constant(0)：")
p = pad2d(x, 1)[0]
print("\n".join("    " + " ".join(f"{v:4.0f}" for v in row) for row in p))
print("\n  pad=1, reflect（镜像，常用于图像增强，不会出现黑边）：")
pr = pad2d(x, 1, mode="reflect")[0]
print("\n".join("    " + " ".join(f"{v:4.0f}" for v in row) for row in pr))

section("3) 尺寸速查表（输入 32×32，kernel=3）")
print(f"{'配置':<26}{'输出尺寸':>10}{'说明':>6}")
for pad, stride in [(0, 1), (1, 1), (2, 1), (0, 2), (1, 2)]:
    H = 32
    out = (H + 2 * pad - 3) // stride + 1
    print(f"  pad={pad}, stride={stride:<14}{out:>8}")

section("4) 空洞卷积（dilation）：不增加参数扩大感受野")
def conv_dilated(x, k, dilation=1):
    """只对单通道 2D 做，返回有效感受野大小说明。"""
    C, H, W = x.shape
    KH, KW = k.shape
    eff_kh = (KH - 1) * dilation + 1
    OH, OW = H - eff_kh + 1, W - eff_kh + 1
    out = np.zeros((OH, OW))
    for i in range(OH):
        for j in range(OW):
            patch = x[0, i:i + eff_kh:dilation, j:j + eff_kh:dilation]
            out[i, j] = np.sum(patch * k)
    return out, eff_kh


kx = np.array([[1.0, 0.0, -1.0]])
img2 = np.zeros((1, 7, 7))
img2[0, :, 3] = 2.0
for dil in [1, 2, 3]:
    _, eff = conv_dilated(img2, kx, dilation=dil)
    print(f"  kernel 1×3, dilation={dil} → 有效核宽 {eff}（感受野 {(eff - 1) * 1 + 1}）")

section("5) 转置卷积：把小特征图『放大』（上采样）")
def conv_transpose2d(x, w, stride=2):
    """最简版本：把每个输入像素乘核后按 stride 铺开（重叠处相加）。"""
    C, H, W = x.shape
    KH, KW = w.shape
    OH, OW = (H - 1) * stride + KH, (W - 1) * stride + KW
    out = np.zeros((OH, OW))
    for i in range(H):
        for j in range(W):
            out[i * stride:i * stride + KH, j * stride:j * stride + KW] += x[0, i, j] * w
    return out[None]


small = np.array([[[1.0, 2.0], [3.0, 4.0]]])
up = conv_transpose2d(small, np.ones((2, 2)), stride=2)
print("  输入 2×2：")
print("\n".join("    " + " ".join(f"{v:4.0f}" for v in row) for row in small[0]))
print("  转置卷积 stride=2, kernel=2×2 全 1 → 输出 4×4：")
print("\n".join("    " + " ".join(f"{v:4.0f}" for v in row) for row in up[0]))
print("  棋盘效应：重叠区域值更大 → 实际用 kernel=4,stride=2 或先插值再卷积来缓解。")

section("6) 结论速记")
print("""
  · 池化 = 降采样 + 平移不变性；现代架构（ConvNeXt / ViT）常用 stride=2 的卷积替代池化
  · Global Average Pool 让分类头不再依赖固定输入尺寸，是 ResNet/现代 CNN 的标准结尾
  · 转置卷积是"可学习的上采样"，UNet / 分割 / 扩散模型的必备件（UNet 在下采样路径）
""")
