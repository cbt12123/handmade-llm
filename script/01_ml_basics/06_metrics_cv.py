"""
06 · 评估指标与交叉验证：怎么诚实地报告"我的模型有多好"
=========================================================
手写实现（不依赖 sklearn）：
  · 混淆矩阵 / 准确率 / 精确率 / 召回率 / F1
  · ROC 曲线与 AUC（梯形面积）
  · k 折交叉验证（手工划分 + 逐折训练）
  · 类别不平衡时为什么"准确率"会骗人
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import section, set_seed, subsection

set_seed(13)


# ----------------------------------------------------------------------------------
# 指标实现
# ----------------------------------------------------------------------------------
def confusion_matrix(y_true, y_pred, num_classes=2):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        cm[t, p] += 1
    return cm


def binary_metrics(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return dict(TP=tp, TN=tn, FP=fp, FN=fn, accuracy=acc, precision=prec, recall=rec, f1=f1)


def roc_curve_auc(y_true, score):
    """ROC：按预测分数从高到低排序，逐个样本当阈值，算 (FPR, TPR)。"""
    order = np.argsort(-score)
    y_sorted = y_true[order]
    n_pos, n_neg = float(np.sum(y_true == 1)), float(np.sum(y_true == 0))
    tpr, fpr = [0.0], [0.0]
    tp = fp = 0.0
    for label in y_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / max(n_pos, 1))
        fpr.append(fp / max(n_neg, 1))
    auc = float(np.trapezoid(np.array(tpr), np.array(fpr)))
    return np.array(fpr), np.array(tpr), auc


# ----------------------------------------------------------------------------------
# 一个可复用的逻辑回归（向量化，含 L2）
# ----------------------------------------------------------------------------------
def train_logreg(X, y, lr=0.5, steps=400, l2=1e-3):
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    w = np.zeros(Xb.shape[1])
    n = X.shape[0]
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(Xb @ w)))
        g = Xb.T @ (p - y) / n
        g[:-1] += l2 * w[:-1]                      # 偏置不加正则
        w -= lr * g
    return w


def predict_logreg(X, w, threshold=0.5):
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    p = 1.0 / (1.0 + np.exp(-(Xb @ w)))
    return (p >= threshold).astype(int), p


# ----------------------------------------------------------------------------------
# 实验数据（不平衡）
# ----------------------------------------------------------------------------------
section("1) 类别不平衡：准确率会骗人")
n_neg, n_pos = 300, 30
X0 = np.random.randn(n_neg, 2) * 1.0
X1 = np.random.randn(n_pos, 2) * 1.0 + np.array([1.6, 1.6])
X = np.vstack([X0, X1])
y = np.hstack([np.zeros(n_neg), np.ones(n_pos)])
X = (X - X.mean(0)) / X.std(0)

w = train_logreg(X, y)
pred, score = predict_logreg(X, w)
m = binary_metrics(y, pred)
print(f"  混淆矩阵 (行=真值, 列=预测):\n{confusion_matrix(y, pred)}")
print(f"  准确率={m['accuracy']:.4f}  精确率={m['precision']:.4f}  召回率={m['recall']:.4f}  F1={m['f1']:.4f}")
print(f"  「全部预测为 0」的准确率 = {(y == 0).mean():.4f}  ← 什么都没学到却很高！")

fpr, tpr, auc = roc_curve_auc(y, score)
print(f"  AUC = {auc:.4f}   （0.5=随机，1.0=完美；与阈值选择无关，是不平衡数据下的更好指标）")

section("2) k 折交叉验证（k=5）：手写实现")
def kfold_indices(n, k, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return np.array_split(idx, k)


folds = kfold_indices(X.shape[0], 5)
accs, aucs = [], []
for i, val_idx in enumerate(folds):
    train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != i])
    w_f = train_logreg(X[train_idx], y[train_idx], steps=300)
    pred_f, score_f = predict_logreg(X[val_idx], w_f)
    mm = binary_metrics(y[val_idx], pred_f)
    _, _, auc_f = roc_curve_auc(y[val_idx], score_f)
    accs.append(mm["accuracy"])
    aucs.append(auc_f)
    print(f"  fold {i}: n_train={len(train_idx):>3} n_val={len(val_idx):>2} "
          f"acc={mm['accuracy']:.4f} auc={auc_f:.4f}")
print(f"  5 折平均 acc={np.mean(accs):.4f} ± {np.std(accs):.4f}, auc={np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

subsection("3) 阈值扫描：精确率-召回率的取舍")
print(f"{'阈值':<8}{'精确率':>10}{'召回率':>10}{'F1':>10}")
for thr in [0.2, 0.35, 0.5, 0.65, 0.8]:
    pred_t, _ = predict_logreg(X, w, threshold=thr)
    mm = binary_metrics(y, pred_t)
    print(f"{thr:<8}{mm['precision']:>10.4f}{mm['recall']:>10.4f}{mm['f1']:>10.4f}")

section("4) 结论速记")
print("""
  · 类别不平衡 → 看 PR 曲线 / AUC / F1，不要只看 accuracy
  · 交叉验证的目的：用小数据得到"模型泛化能力"的低方差估计；LLM 时代因为训练太贵，
    通常退化为"固定的 held-out 验证集"，但思想一致
  · LLM 的评测指标沿用了这些概念：BLEU/ROUGE≈精确率型， perplexity≈交叉熵型
""")
