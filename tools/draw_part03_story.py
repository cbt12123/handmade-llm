"""Original figures for the learning-card story, generated from local digits."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
from sklearn.datasets import load_digits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'images'
plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                     'axes.unicode_minus': False, 'font.size': 11})

def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f'part03-story-{name}.png', dpi=160)
    plt.close(fig)

def box(ax, xy, width, height, label, color='#e7efff'):
    ax.add_patch(FancyBboxPatch(xy, width, height, boxstyle='round,pad=.025',
                              facecolor=color, edgecolor='#6085ad'))
    ax.text(xy[0] + width/2, xy[1] + height/2, label, ha='center', va='center')

data = load_digits()
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
index = np.flatnonzero(data.target == 3)[0]
axes[0].imshow(data.images[index], cmap='gray', vmin=0, vmax=16)
axes[0].set_title('学习卡的手写编号：3'); axes[0].axis('off')
for ax in axes[1:]: ax.axis('off'); ax.set(xlim=(0, 1), ylim=(0, 1))
box(axes[1], (.05, .45), .9, .35, '反馈：这道题还没理解')
axes[1].text(.5, .2, '第12章：文字 → 复习状态', ha='center')
box(axes[2], (.03, .45), .94, .35, '取卡 2 4 7 9 1\n复核 1 9 7 4 2', '#e5f4eb')
axes[2].text(.5, .2, '第13章：受控序列生成', ha='center')
save(fig, 'overview')

fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
for ax, digit in zip(axes[:2], [3, 8]):
    image = data.images[np.flatnonzero(data.target == digit)[0]]
    ax.imshow(image, cmap='gray', vmin=0, vmax=16)
    ax.set_title(f'真实数字 {digit} → 二分类标签 {int(digit == 8)}'); ax.axis('off')
axes[2].axis('off'); axes[2].set(xlim=(0, 1), ylim=(0, 1))
box(axes[2], (.05, .45), .9, .4, '64个像素 → 16个隐藏值\n→ 1个logit → 属于8的概率')
axes[2].text(.5, .18, '第9章：只研究3与8\n第10章：恢复完整10类任务', ha='center')
save(fig, 'binary')

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
image = data.images[index]; axes[0].imshow(image, cmap='gray', vmin=0, vmax=16)
axes[0].add_patch(Rectangle((1.5, 1.5), 3, 3, fill=False, edgecolor='#ffb000', linewidth=3))
axes[0].set(title='卷积读取相邻的3×3窗口', xticks=range(8), yticks=range(8))
axes[1].imshow(image.reshape(1, 64), cmap='gray', vmin=0, vmax=16, aspect='auto')
axes[1].set(title='展平后依然保留数值，但模型没有内置邻接结构', xlabel='特征编号', yticks=[])
save(fig, 'image-layout')

fig, ax = plt.subplots(figsize=(11, 3)); ax.axis('off'); ax.set(xlim=(0, 11), ylim=(0, 3))
for i, char in enumerate('我还不会'):
    box(ax, (.15 + i*1.2, 1.9), 1, .65, char)
    box(ax, (.15 + i*1.2, .75), 1, .65, f'编号 id{i}', '#e5f4eb')
    ax.annotate('', xy=(.65+i*1.2, 1.43), xytext=(.65+i*1.2, 1.87), arrowprops={'arrowstyle':'->'})
box(ax, (5.7, .85), 2.1, 1.3, '按顺序读取\nGRU状态')
box(ax, (8.5, .85), 2.1, 1.3, '标签0\n建议复习')
ax.annotate('', xy=(5.6, 1.5), xytext=(4.8, 1.5), arrowprops={'arrowstyle':'->'})
ax.annotate('', xy=(8.4, 1.5), xytext=(7.9, 1.5), arrowprops={'arrowstyle':'->'})
ax.text(.15, .1, '词表编号由训练文字决定；图中的 id0…id3 是占位说明，不是固定编号。')
save(fig, 'text-order')

fig, ax = plt.subplots(figsize=(11, 3.2)); ax.axis('off'); ax.set(xlim=(0, 8), ylim=(0, 3))
inputs = ['数字2', '数字4', '数字7', 'SEP', '数字7', '数字4', '数字2']
targets = ['忽略', '忽略', '忽略', '数字7', '数字4', '数字2', 'EOS']
for i, (input_text, target) in enumerate(zip(inputs, targets)):
    box(ax, (i+.12, 1.65), .8, .65, input_text)
    box(ax, (i+.12, .5), .8, .65, target, '#eeeeee' if i<3 else '#e5f4eb')
    ax.annotate('', xy=(i+.52, 1.2), xytext=(i+.52, 1.6), arrowprops={'arrowstyle':'->'})
ax.text(7.25, 1.97, '输入', va='center'); ax.text(7.25, .82, '目标', va='center')
ax.text(.12, 2.65, '教学样例取3张卡；实际基础实验训练长度为5。')
ax.text(.12, .05, '从SEP所在位置开始计入目标损失；每个位置只读自己及之前的输入。')
save(fig, 'teacher-forcing')
