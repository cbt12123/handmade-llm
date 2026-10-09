"""Original teaching figures; no images reproduced from the reference book."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
arrays = [np.arange(1, 17).reshape(4, 4), np.array([[1, 0], [0, -1]]), np.full((3, 3), -5)]
titles = ['输入：4×4像素', '共享卷积核：2×2', '输出：每格都是−5']
for ax, arr, title in zip(axes, arrays, titles):
    ax.imshow(arr, cmap='Blues', vmin=-5, vmax=16)
    for (row, col), value in np.ndenumerate(arr):
        ax.text(col, row, str(value), ha='center', va='center')
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
axes[0].add_patch(Rectangle((-.5, -.5), 2, 2, fill=False, edgecolor='orange', linewidth=3))
fig.suptitle('先算一格：1×1 + 2×0 + 5×0 + 6×(−1) = −5')
fig.tight_layout(); fig.savefig(ROOT/'images'/'part03-hand-cnn-window.png', dpi=160); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
arrays = [np.array([[1, 3], [2, 0]]), np.array([[0, 7], [0, 0]]), np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])]
titles = ['池化：最大值3的位置被记住', '上游梯度7：返回最大值位置', '重叠窗口：输入梯度需要累加']
for ax, arr, title in zip(axes, arrays, titles):
    ax.imshow(arr, cmap='Oranges')
    for (row, col), value in np.ndenumerate(arr):
        ax.text(col, row, str(value), ha='center', va='center')
    ax.set_title(title, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('右图条件：2×2全1卷积核，2×2输出的上游梯度也全为1')
fig.tight_layout(); fig.savefig(ROOT/'images'/'part03-hand-cnn-backward.png', dpi=160); plt.close(fig)
