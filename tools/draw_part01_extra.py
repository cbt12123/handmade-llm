"""Generate the diagrams for chapters 2 and 3."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from draw_part01 import canvas, box, arrow, save, INK, GREEN

fig, ax = canvas("相对路径从工作目录出发")
box(ax,.5,2,3.4,1.1,"脚本位置\nG:/study/code/main.py")
box(ax,6,2,4,1.1,"终端工作目录\nG:/study")
box(ax,3,.3,5,1.1,'Path("data.csv")\n查找 G:/study/data.csv',GREEN)
arrow(ax,(8,1.9),(6,1.5))
ax.text(.5,3.45,"文件在哪里 ≠ 从哪里运行文件",fontsize=14,color=INK)
save(fig,"paths")

fig, ax = canvas("清洗顺序：每一步都有明确规则")
labels=["原始记录\n6 行","清理空白\n检查缺失","转换数字\n检查范围","按记录去重\n保留 2 行"]
for i,label in enumerate(labels):
    x=.4+i*2.7
    box(ax,x,1.9,2.1,1.1,label,GREEN if i==3 else "#e5efff")
    if i<3: arrow(ax,(x+2.2,2.45),(x+2.6,2.45))
ax.text(.6,.65,"剔除记录要保留原因：缺失、非数字、超范围、重复。",fontsize=14,color=INK)
save(fig,"cleaning")

fig, axes=plt.subplots(2,2,figsize=(10,7))
axes[0,0].plot([1,2,3,4],[10,20,15,30],marker="o")
axes[0,0].set(title="Line: change over time",xlabel="Day",ylabel="Minutes")
axes[0,1].bar(["A","B","C"],[3,5,2])
axes[0,1].set(title="Bar: compare categories",xlabel="Group",ylabel="Count")
scores=[50,60,65,70,80,85]
axes[1,0].scatter([1,2,2,3,4,5],scores)
axes[1,0].set(title="Scatter: relationship",xlabel="Hours",ylabel="Score")
axes[1,1].hist(scores,bins=[40,60,80,100],edgecolor="white")
axes[1,1].set(title="Histogram: distribution",xlabel="Score",ylabel="Count")
fig.tight_layout()
save(fig,"chart-types")

scores=np.array([[80,90],[60,70],[100,80]])
fig,ax=plt.subplots(figsize=(5,4))
im=ax.imshow(scores,cmap="Blues",vmin=0,vmax=100)
ax.set_xticks([0,1],labels=["Math","English"])
ax.set_yticks([0,1,2],labels=["Student A","Student B","Student C"])
for r in range(3):
    for c in range(2):
        ax.text(c,r,str(scores[r,c]),ha="center",va="center",color="black" if scores[r,c]<80 else "white")
fig.colorbar(im,ax=ax,label="Score")
fig.tight_layout()
save(fig,"score-heatmap")
print("Generated 4 additional teaching images.")
