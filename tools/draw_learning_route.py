"""Course prerequisite map, drawn with base Matplotlib."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
fig, ax = plt.subplots(figsize=(12, 6))
ax.set(xlim=(0, 12), ylim=(0, 6))
ax.axis("off")
ax.text(6, 5.55, "课程主线与按需补读", ha="center", fontsize=22, weight="bold")

def box(x, y, text, color="#eff6ff"):
    ax.add_patch(FancyBboxPatch((x, y), 2.2, .9, boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor="#94a3b8"))
    ax.text(x+1.1, y+.45, text, ha="center", va="center", fontsize=12, linespacing=1.5)

for x, label in [(0.5, "第一部分\nPython 与数组"), (3.4, "第二部分\n数学起点"),
                 (6.3, "第三部分\n训练与模型评价"), (9.2, "第四部分\n模型加载与部署")]:
    box(x, 3.8, label)
for start in [2.7, 5.6, 8.5]:
    ax.annotate("", xy=(start+.7, 4.25), xytext=(start, 4.25),
                arrowprops={"arrowstyle": "->", "color": "#475569", "lw": 1.5})
box(7.0, 1.8, "第五部分\n课程学习 Agent", "#f0fdf4")
box(9.4, 1.8, "第六部分\nNVIDIA 算子", "#f0fdf4")
for end in [8.1, 10.5]:
    ax.annotate("", xy=(end, 2.8), xytext=(10.3, 3.65),
                arrowprops={"arrowstyle": "->", "color": "#475569", "lw": 1.5})
ax.text(8.9, 1.25, "两条应用分支独立；算子不以 Agent 为前置", ha="center", fontsize=11)
ax.text(4.1, 2.6, "遇到具体缺口再补读", ha="center", fontsize=13, weight="bold")
ax.text(4.1, 1.75, "工程 Python：状态、参数、异常、接口\n数学速查：索引、形状、梯度、概率", ha="center",
        fontsize=12, linespacing=1.8, color="#475569")
ax.text(6, .45, "先完成入口自测；新章节要教的内容不设为入门门槛", ha="center", fontsize=13)
output = Path(__file__).resolve().parents[1]/"images"/"learning-route.png"
fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(output)
