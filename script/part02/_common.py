from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "part02"
OUT.mkdir(parents=True, exist_ok=True)

def write_json(name, data):
    path = OUT / name
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(path)

def transform_figure():
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
    theta = np.pi / 4
    matrices = [np.eye(2), np.diag([2, .5]), np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.diag([1, 0])]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    for ax, matrix, title in zip(axes, matrices, ["Original", "Scale", "Rotate 45 degrees", "Project to x-axis"]):
        transformed = points @ matrix.T
        ax.plot(transformed[:, 0], transformed[:, 1], "o-")
        ax.set(title=title, xlim=(-1, 2.5), ylim=(-.5, 2), xlabel="x", ylabel="y")
        ax.set_aspect("equal")
        ax.grid(alpha=.25)
    fig.tight_layout()
    return fig

def graph_figure():
    positions = {"A": (0, 1), "B": (1, 2), "C": (2, 1), "D": (3, 2)}
    edges = [("A", "B", 2), ("A", "C", 5), ("B", "C", 1), ("B", "D", 4), ("C", "D", 1)]
    fig, ax = plt.subplots(figsize=(8, 4))
    for source, target, cost in edges:
        start, end = positions[source], positions[target]
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", shrinkA=20, shrinkB=20, lw=2, color="#546d91"))
        ax.text((start[0]+end[0])/2, (start[1]+end[1])/2+.12, str(cost), ha="center", bbox=dict(facecolor="white", edgecolor="none"))
    for name, (x, y) in positions.items():
        ax.scatter(x, y, s=900, color="#e2efff", edgecolor="#597399", zorder=3)
        ax.text(x, y, name, ha="center", va="center", zorder=4)
    ax.set(xlim=(-.4, 3.4), ylim=(.5, 2.6), title="Weighted DAG: A -> B -> C -> D costs 4")
    ax.axis("off")
    fig.tight_layout()
    return fig
