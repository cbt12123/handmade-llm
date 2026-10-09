"""Shared paths, split conventions, reports and plots; no torch dependency."""
from pathlib import Path
import json
import platform
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "part03"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42

def split_indices(y):
    """60% train, 20% validation, 20% test; stratified and reproducible."""
    indices = np.arange(len(y))
    development, test = train_test_split(indices, test_size=.2, stratify=y, random_state=SEED)
    train, valid = train_test_split(development, test_size=.25, stratify=y[development], random_state=SEED)
    return train, valid, test

def metrics(y, prediction):
    return {"accuracy":float(accuracy_score(y,prediction)), "macro_f1":float(f1_score(y,prediction,average="macro",zero_division=0)), "confusion_matrix":confusion_matrix(y,prediction).tolist()}

def report(name, data):
    data["environment"] = {"python":platform.python_version(), "numpy":np.__version__}
    path = OUT / name
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(path)

def curves(history, name):
    fig,axes = plt.subplots(1,2,figsize=(10,4))
    for key in ["train_loss","valid_loss"]:
        if key in history: axes[0].plot(history[key],label=key)
    for key in ["train_acc","valid_acc"]:
        if key in history: axes[1].plot(history[key],label=key)
    axes[0].set(title="Loss",xlabel="Epoch / logged step")
    axes[1].set(title="Accuracy",xlabel="Epoch / logged step",ylim=(0,1.05))
    for ax in axes: ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(OUT/name,dpi=150); plt.close(fig)
