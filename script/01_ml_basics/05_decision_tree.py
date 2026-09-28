"""
05 · 手写决策树（CART）：不用任何微积分的模型
==============================================
梯度下降不是唯一的路。决策树回答另一个问题："用哪个特征、在哪个阈值切一刀，
能让左右两堆样本最纯？"

纯度指标（分类）：
    Gini(p) = 1 − Σ pₖ²          （0=纯，0.5=二分类最杂）
    Entropy(p) = −Σ pₖ log pₖ
切分收益：
    Gain =  impurity(父) − (n_L/n)·impurity(左) − (n_R/n)·impurity(右)

树的递归就三行：找最优切分 → 切两半 → 对两半递归。
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import section, set_seed, subsection

set_seed(5)


# ----------------------------------------------------------------------------------
# 纯度 / 不纯度
# ----------------------------------------------------------------------------------
def gini(labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    p = np.bincount(labels.astype(int)) / labels.size
    return float(1.0 - np.sum(p**2))


def entropy(labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    p = np.bincount(labels.astype(int)) / labels.size
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def variance(values: np.ndarray) -> float:
    return float(np.var(values)) if values.size else 0.0


def best_split(X: np.ndarray, y: np.ndarray, task: str):
    """暴力搜索最优 (特征, 阈值)：对每个特征的每个候选阈值试一遍。

    这是 O(n·d·log n) 的朴素实现，真实库会用分位数加速，但思想完全一致。
    """
    n, d = X.shape
    parent = gini(y) if task == "clf" else variance(y)
    best = {"gain": -1.0, "feat": None, "thr": None}
    for feat in range(d):
        thresholds = np.unique(X[:, feat])
        if thresholds.size > 32:                       # 候选阈值太多时抽稀，加速
            thresholds = np.quantile(X[:, feat], np.linspace(0, 1, 32))
        for thr in thresholds:
            left = X[:, feat] <= thr
            if left.all() or (~left).all():
                continue
            yl, yr = y[left], y[~left]
            impurity = gini(yl) if task == "clf" else variance(yl)
            impurity_r = gini(yr) if task == "clf" else variance(yr)
            child = (yl.size * impurity + yr.size * impurity_r) / n
            gain = parent - child
            if gain > best["gain"]:
                best = {"gain": gain, "feat": feat, "thr": float(thr)}
    return best


class Node:
    """一个节点要么是内部节点（有切分规则），要么是叶子（有预测值）。"""

    def __init__(self, feat=None, thr=None, left=None, right=None, value=None, n=0, imp=0.0):
        self.feat, self.thr, self.left, self.right = feat, thr, left, right
        self.value, self.n, self.imp = value, n, imp

    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTree:
    def __init__(self, task="clf", max_depth=4, min_samples_split=2):
        self.task = task
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.n_splits = 0

    def fit(self, X, y):
        self.root = self._build(X, y, depth=0)
        return self

    def _build(self, X, y, depth):
        if self.task == "clf":
            value = np.bincount(y.astype(int)).argmax()      # 多数投票
            imp = gini(y)
        else:
            value = float(y.mean())
            imp = variance(y)

        if depth >= self.max_depth or y.size < self.min_samples_split or imp < 1e-12:
            return Node(value=value, n=y.size, imp=imp)

        split = best_split(X, y, self.task)
        if split["feat"] is None or split["gain"] <= 1e-12:
            return Node(value=value, n=y.size, imp=imp)

        left_mask = X[:, split["feat"]] <= split["thr"]
        self.n_splits += 1
        return Node(
            feat=split["feat"], thr=split["thr"],
            left=self._build(X[left_mask], y[left_mask], depth + 1),
            right=self._build(X[~left_mask], y[~left_mask], depth + 1),
            n=y.size, imp=imp,
        )

    def _predict_one(self, node, x):
        while not node.is_leaf():
            node = node.left if x[node.feat] <= node.thr else node.right
        return node.value

    def predict(self, X):
        return np.array([self._predict_one(self.root, x) for x in X])

    def print_tree(self, node=None, depth=0, prefix="root"):
        node = node or self.root
        pad = "    " * depth
        if node.is_leaf():
            v = node.value if self.task == "clf" else round(node.value, 3)
            print(f"{pad}{prefix}: 叶子 → 预测 {v} (n={node.n}, 不纯度={node.imp:.3f})")
        else:
            print(f"{pad}{prefix}: x[{node.feat}] <= {node.thr:.3f} ? (n={node.n}, 不纯度={node.imp:.3f})")
            self.print_tree(node.left, depth + 1, "├─ 左")
            self.print_tree(node.right, depth + 1, "└─ 右")


# ----------------------------------------------------------------------------------
# 分类实验：异或风格的四象限数据（决策树最擅长的形态）
# ----------------------------------------------------------------------------------
section("1) 分类树：四象限数据")
X = np.random.rand(200, 2) * 4 - 2
y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)          # XOR
tree = DecisionTree(task="clf", max_depth=3).fit(X, y)
tree.print_tree()
print(f"\n切分次数 = {tree.n_splits}")
print(f"训练集准确率 = {np.mean(tree.predict(X) == y):.4f}")

section("2) 回归树：拟合 y = sin(3x) + 噪声")
xs = np.sort(np.random.rand(80) * 6 - 3).reshape(-1, 1)
ys = np.sin(3 * xs).ravel() + np.random.randn(80) * 0.15
for depth in [1, 3, 10]:
    t = DecisionTree(task="reg", max_depth=depth).fit(xs, ys)
    mse = float(np.mean((t.predict(xs) - ys) ** 2))
    print(f"  max_depth={depth:>2}  训练集 MSE={mse:.4f}  叶子数={t.n_splits + 1}")

print("\n观察：深度越大训练误差越小，但那只是把噪声也记下来了——这就是过拟合。")

subsection("3) 决策树 vs 神经网络：什么时候选谁")
print("""
  决策树：可解释、无需标准化、能处理混合类型特征、天然处理非线性；
          但不擅长外推（预测值永远在训练集范围内），高维稀疏数据（文本）表现差。
  神经网络：可微分 → 能用梯度下降端到端训练，天然适合高维稠密/序列/图像数据。
  现实中的 LLM 全是神经网络路线，但树的集成（GBDT/XGBoost）仍是表格数据的王者。
  注：LLM 里的 MoE 路由（第 6 章）在思想上是"用可微分的软决策树做专家选择"。
""")
