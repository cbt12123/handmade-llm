"""
09 · 真实生态：用 scikit-learn 重做第 1 章的全部模型
=====================================================
手写是为了理解原理，真实项目要用成熟库。本节把第 1 章的手写实现与 sklearn 一一对照，
并回答"什么时候该用库、什么时候该自己写"。

对照清单：
  手写线性回归（闭式解）   ↔ sklearn.linear_model.LinearRegression
  手写逻辑回归             ↔ sklearn.linear_model.LogisticRegression
  手写 MLP                 ↔ sklearn.neural_network.MLPClassifier
  手写 CART 决策树         ↔ sklearn.tree.DecisionTreeClassifier
  手写指标/CV              ↔ sklearn.metrics / model_selection
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import check_close, require_module, section, set_seed, subsection, timer

set_seed(0)

sklearn = require_module("sklearn", "本章手写实现不依赖它，可照常学习",
                         pip_name="scikit-learn")
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# ----------------------------------------------------------------------------------
section("1) 线性回归：手写闭式解 vs sklearn")
X, y = make_regression(n_samples=400, n_features=3, noise=8.0, random_state=0)

# 手写：w = (XᵀX)⁻¹Xᵀy（含截距：给 X 补一列 1）
Xb = np.c_[np.ones(len(X)), X]
w_manual = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)

lr = LinearRegression().fit(X, y)
print(f"  手写: intercept={w_manual[0]:.4f}  coef={np.round(w_manual[1:], 4)}")
print(f"  sklearn: intercept={lr.intercept_:.4f}  coef={np.round(lr.coef_, 4)}")
check_close(np.r_[lr.intercept_, lr.coef_], w_manual, tol=1e-6,
            name="手写闭式解 vs sklearn.LinearRegression")

# ----------------------------------------------------------------------------------
section("2) 逻辑回归：手写梯度下降 vs sklearn（带 L2）")
Xc, yc = make_classification(n_samples=600, n_features=8, n_informative=4,
                             n_classes=2, random_state=0)


def logreg_manual(X, y, lr_rate=0.5, steps=1500, l2=1e-3):
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        dz = (p - y) / n
        gw = X.T @ dz + l2 * w                      # ← L2 就多这一项
        w -= lr_rate * gw
        b -= lr_rate * dz.sum()
    return w, b


w_m, b_m = logreg_manual(Xc, yc)
p_m = (1 / (1 + np.exp(-(Xc @ w_m + b_m))) >= 0.5).astype(int)

# sklearn 默认就带 L2 正则（C 是正则强度的倒数）
clf = LogisticRegression(max_iter=2000, C=100.0).fit(Xc, yc)
p_s = clf.predict(Xc)
print(f"  手写准确率 = {accuracy_score(yc, p_m):.4f}    sklearn 准确率 = {accuracy_score(yc, p_s):.4f}")
print(f"  两者预测一致率 = {(p_m == p_s).mean():.4f}")
print("  差异来源：sklearn 用的是 L-BFGS/牛顿法 + 更强正则，我们用的是朴素 SGD。")

# ----------------------------------------------------------------------------------
section("3) 标准化的必要性（造一个量纲悬殊的数据集）")
from sklearn.linear_model import SGDClassifier

Xs, ys = make_classification(n_samples=800, n_features=5, n_informative=3, random_state=1)
Xs = Xs * np.array([1000.0, 0.001, 1.0, 250.0, 0.5])       # ← 各维量纲差 6 个数量级
print(f"  各维标准差: {np.round(Xs.std(axis=0), 3)}")

Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=0.25, random_state=0, stratify=ys)
scaler = StandardScaler().fit(Xtr)                          # ← 只在训练集上 fit！

for tag, fit_X, eval_X in [("不标准化", Xtr, Xte),
                           ("标准化后", scaler.transform(Xtr), scaler.transform(Xte))]:
    m = SGDClassifier(loss="log_loss", max_iter=2000, tol=1e-4, random_state=0).fit(fit_X, ytr)
    print(f"  {tag:<8}(SGD)  : 测试准确率 = {accuracy_score(yte, m.predict(eval_X)):.4f}")

m_lr = LogisticRegression(max_iter=2000).fit(scaler.transform(Xtr), ytr)
print(f"  标准化后(L-BFGS): 测试准确率 = "
      f"{accuracy_score(yte, m_lr.predict(scaler.transform(Xte))):.4f}")
print("""
  结论：基于梯度的优化器对量纲极其敏感（SGD 尤其明显）；
  树模型（决策树/随机森林）则完全不受影响 —— 这就是为什么表格数据里 GBDT 省心。
  ⚠️ 常见错误：先在整个数据集上 fit 再切分 → 数据泄漏（测试集统计信息进了训练集）。
""")

# ----------------------------------------------------------------------------------
section("4) 四个模型横向对比 + 交叉验证")
models = {
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "MLPClassifier": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800, random_state=0),
    "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=0),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=0),
}
Xa, Xb_, ya, yb = train_test_split(Xc, yc, test_size=0.25, random_state=0, stratify=yc)
sc2 = StandardScaler().fit(Xa)                           # 在 8 维数据的训练集上 fit
pipe_X = sc2.transform(Xc)                               # 指标展示用（全量）
print(f"  {'模型':<22}{'5折CV准确率':>14}{'全量F1':>10}{'AUC':>8}")
for name, m in models.items():
    cv = cross_val_score(m, pipe_X, yc, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    m.fit(pipe_X, yc)                                        # 指标展示用：全量 fit
    pred = m.predict(pipe_X)
    prob = m.predict_proba(pipe_X)[:, 1] if hasattr(m, "predict_proba") else pred
    print(f"  {name:<22}{cv.mean():>14.4f}{f1_score(yc, pred):>10.4f}"
          f"{roc_auc_score(yc, prob):>8.4f}")

section("5) 决策树规则的可解释性（这是它至今好用的原因）")
tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(pipe_X, yc)
print(export_text(tree, max_depth=3, feature_names=[f"f{i}" for i in range(Xc.shape[1])])[:600])

section("6) 什么时候自己写，什么时候用库")
print("""
  用库（sklearn / xgboost / lightgbm）：
    · 表格数据、基线模型、快速验证想法、需要可解释性与稳健的默认参数
    · 工业级实现：数值稳定、并行、处理缺失值/类别特征/不平衡
  自己写：
    · 学习原理（本章的目的）
    · 需要自定义损失/结构/梯度的研究场景 → 用 PyTorch（第 4 章之后）
    · 要部署到特殊硬件 → 自己写算子（第 7 章）
  经验顺序：先跑 sklearn 基线 → 再用深度学习看能不能显著超过它。
  很多时候，一个调好的 GBDT 就赢了。
""")
