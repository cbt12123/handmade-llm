"""Draw the environment map with Matplotlib; no network required."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
fig, ax = plt.subplots(figsize=(12, 6))
ax.set(xlim=(0, 12), ylim=(0, 6))
ax.axis("off")
ax.text(6, 5.65, "按当前学习内容选择运行环境", ha="center", fontsize=22, weight="bold")

def box(x, y, width, height, text, color):
    ax.add_patch(FancyBboxPatch((x, y), width, height,
                               boxstyle="round,pad=0.12", facecolor=color,
                               edgecolor="#94a3b8", linewidth=1.2))
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=13,
            linespacing=1.7)

box(.4, 4.3, 11.2, .55, "VS Code：编辑代码与文档；终端：选择在哪里执行命令", "#eef2ff")
box(.4, 1, 5.3, 2.6, "Windows · conda\nbase：Python / 数学 / Agent 客户端\nhandmade-ml：机器学习（CPU）\nhandmade-llm：本地模型与 HTTP", "#eff6ff")
box(6.3, 1, 5.3, 2.6, "WSL 2 · Docker · NVIDIA GPU\nvLLM 镜像：服务 / Torch / Triton\nCUDA devel 镜像：编译与内存检查\n容器内使用独立的 Python", "#f0fdf4")
ax.annotate("HTTP 请求", xy=(6.2, 2.2), xytext=(5.8, 2.2), ha="center", fontsize=10,
            arrowprops={"arrowstyle": "->", "color": "#475569"}, rotation=90)
ax.text(6, .35, "代码与模型通过目录挂载进入容器；同一个环境检查脚本在对应 Python 中执行",
        ha="center", fontsize=12, color="#475569")
output = Path(__file__).resolve().parents[1] / "images" / "environment-overview.png"
fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(output)
