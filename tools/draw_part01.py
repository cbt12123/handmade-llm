"""Generate original teaching diagrams with Matplotlib; run from any directory."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "images"
OUT.mkdir(exist_ok=True)
available = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.sans-serif"] = [n for n in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"] if n in available]
plt.rcParams["axes.unicode_minus"] = False
BLUE, INK, GREEN = "#e5efff", "#23354b", "#e2f5ed"

def canvas(title, size=(11, 4.5)):
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor("#fafcff")
    ax.set(xlim=(0, 11), ylim=(0, 4.5))
    ax.axis("off")
    ax.text(.35, 4.12, title, fontsize=19, weight="bold", color=INK)
    return fig, ax

def box(ax, x, y, w, h, label, color=BLUE, fontsize=13):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12", facecolor=color, edgecolor="#a5b9cf"))
    ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=fontsize, color=INK)

def arrow(ax, start, end, label=None):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.8, color="#536e90"))
    if label:
        ax.text((start[0]+end[0])/2, (start[1]+end[1])/2+.2, label, ha="center", fontsize=11, color=INK)

def save(fig, name):
    fig.savefig(OUT / f"part01-{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

fig, ax = canvas("一段代码怎样运行？")
box(ax,.5,2,2.7,1,"VS Code\n编辑并保存 hello.py")
box(ax,4,2,2.7,1,"Python 解释器\n执行文件中的代码")
box(ax,7.5,2,2.7,1,"终端\n显示程序输出",GREEN)
arrow(ax,(3.3,2.5),(3.9,2.5))
arrow(ax,(6.8,2.5),(7.4,2.5))
box(ax,3.4,.35,4, .8,"conda base：解释器 + 已安装的库",GREEN)
arrow(ax,(5.4,1.2),(5.4,1.9))
ax.text(.5,3.5,"运行命令输入在终端：python hello.py",fontsize=13,color=INK)
save(fig,"environment")

fig, ax = canvas("条件判断：从上到下找到第一个成立的分支")
box(ax,.5,2.25,2.5,.8,"score ≥ 90？")
box(ax,4,2.25,2.5,.8,"score ≥ 60？")
box(ax,7.5,2.25,2.6,.8,"需要继续练习",GREEN)
arrow(ax,(3.1,2.65),(3.9,2.65),"否")
arrow(ax,(6.6,2.65),(7.4,2.65),"否")
box(ax,.5,.45,2.5,.8,"优秀",GREEN)
box(ax,4,.45,2.5,.8,"通过",GREEN)
arrow(ax,(1.75,2.15),(1.75,1.35),"是")
arrow(ax,(5.25,2.15),(5.25,1.35),"是")
save(fig,"branch")

fig, ax = canvas("切片 values[1:4]：包含开始，不包含结束")
for i, val in enumerate([10,20,30,40,50]):
    x=1+i*1.8
    box(ax,x,1.7,1.5,.85,str(val),GREEN if 1<=i<4 else BLUE,18)
    ax.text(x+.75,2.9,f"索引 {i}",ha="center",fontsize=13,color=INK)
ax.text(2.8,.75,"保留索引 1、2、3 → [20, 30, 40]",fontsize=16,color=INK)
save(fig,"slice")

fig, ax = canvas("函数：参数传入，return 把结果传回")
box(ax,.5,1.65,2.2,1.2,"输入\na=2，b=3")
box(ax,4,1.65,2.5,1.2,"add(a, b)\n计算 a + b")
box(ax,7.8,1.65,2.2,1.2,"返回值\n5",GREEN)
arrow(ax,(2.8,2.25),(3.9,2.25))
arrow(ax,(6.6,2.25),(7.7,2.25),"return")
ax.text(.6,.65,"answer = add(2, 3)    →    answer 得到 5",fontsize=15,color=INK)
save(fig,"function")

fig, ax = canvas("二维数组：shape=(2, 2)，行是样本，列是特征")
for r,row in enumerate([[170,60],[180,75]]):
    for c,value in enumerate(row):
        box(ax,3+c*1.8,2.15-r*1.15,1.5,.85,str(value),fontsize=18)
ax.text(3.75,3.25,"身高",ha="center",fontsize=14,color=INK)
ax.text(5.55,3.25,"体重",ha="center",fontsize=14,color=INK)
ax.text(.6,2.55,"第 0 人",fontsize=14,color=INK)
ax.text(.6,1.4,"第 1 人",fontsize=14,color=INK)
arrow(ax,(7.5,3),(7.5,.9))
ax.text(7.8,1.8,"axis=0\n汇总行 → 每列均值",fontsize=12,color=INK)
arrow(ax,(3,.45),(6.3,.45))
ax.text(3,.05,"axis=1：汇总列 → 每行均值",fontsize=12,color=INK)
save(fig,"array")

fig, ax = canvas("广播：同一组列偏移，作用到两行")
box(ax,.5,1.65,2.4,1.3,"data：形状 (2, 3)\n1   2   3\n4   5   6")
ax.text(3.2,2.2,"+",fontsize=25,color=INK)
box(ax,4,1.65,2.3,1.3,"offset：形状 (3,)\n10   20   30")
arrow(ax,(6.4,2.3),(7.3,2.3))
box(ax,7.5,1.65,2.8,1.3,"结果：形状 (2, 3)\n11   22   33\n14   25   36",GREEN)
ax.text(.7,.7,"从右对齐维度：3 与 3 匹配；缺少的维度按 1 处理。",fontsize=14,color=INK)
save(fig,"broadcast")

fig, ax = plt.subplots(figsize=(8,4.5))
ax.plot([1,2,3,4],[10,20,15,30],marker="o")
ax.set(xlabel="Day",ylabel="Study time (minutes)",title="Study time",xticks=[1,2,3,4])
ax.grid(alpha=.3)
fig.tight_layout()
save(fig,"study-time")
print(f"Generated 7 images in {OUT}")
