"""全仓库脚本共用的小工具库。

设计原则：
1. **零硬依赖**：只依赖标准库 + numpy；torch / matplotlib / CUDA 都是"可选增强"。
2. **可离线运行**：所有章节都使用合成数据或内置微型语料，不联网下载数据集。
3. **输出可读**：每个脚本都能独立 `python3 xxx.py` 跑通并打印结论。

用法（每个章节脚本开头这样写）：::

    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))  # 找到 script/
    from common_utils import set_seed, section, has_torch, device
"""

from __future__ import annotations

import math
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------------------
# 随机性控制
# --------------------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """固定 python / numpy / torch 的随机种子，保证实验结果可复现。"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # numpy 可能没装（虽然所有章节都需要它）
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover
        pass


# --------------------------------------------------------------------------------------
# 可选依赖探测
# --------------------------------------------------------------------------------------


def has_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def has_module(name: str) -> bool:
    """探测任意第三方库是否可用（transformers / peft / sklearn / ...）。"""
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def require_module(name: str, who: str, pip_name: str = ""):
    """缺少依赖时给出清晰的 [SKIP]，不抛异常。"""
    import importlib

    if not has_module(name):
        skip(f"需要 `{pip_name or name}`（pip install {pip_name or name}）。{who}")
    return importlib.import_module(name)


def has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def has_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401

        return True
    except Exception:
        return False


def has_triton() -> bool:
    try:
        import triton  # noqa: F401

        return True
    except Exception:
        return False


def device(prefer_cuda: bool = True) -> str:
    """返回 'cuda' 或 'cpu' 字符串（不 import torch 到全局，避免无 torch 环境报错）。"""
    return "cuda" if (prefer_cuda and has_cuda()) else "cpu"


def torch_or_none():
    """返回 torch 模块或 None。"""
    try:
        import torch

        return torch
    except Exception:
        return None


def require_torch(script_name: str = "本脚本"):
    """需要 torch 时调用：没有就给出清晰的安装提示并正常退出（退出码 0 表示"跳过"）。"""
    torch = torch_or_none()
    if torch is None:
        print(f"[SKIP] {script_name} 需要 PyTorch：pip install torch")
        raise SystemExit(0)
    return torch


# --------------------------------------------------------------------------------------
# 打印 / 计时 / 画图
# --------------------------------------------------------------------------------------


def section(title: str, width: int = 72) -> None:
    """打印分节标题，让脚本输出像一份可读性强的讲义。"""
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def subsection(title: str, width: int = 72) -> None:
    print("\n" + "-" * width)
    print(title)
    print("-" * width)


@contextmanager
def timer(name: str = "耗时"):
    """with timer("前向传播"): ...   -> 自动打印耗时。"""
    t0 = time.perf_counter()
    yield
    dt = (time.perf_counter() - t0) * 1000
    print(f"[{name}] {dt:.2f} ms" if dt < 1000 else f"[{name}] {dt / 1000:.2f} s")


def save_fig(fig, path: str | Path, dpi: int = 110) -> Optional[Path]:
    """有 matplotlib 就保存图片，没有就静默跳过（保证脚本在纯 CPU 环境也能跑）。"""
    if not has_matplotlib():
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[fig] 已保存图片 -> {path}")
    return path


_ASCII_RAMP = " .:-=+*#%@"


def ascii_heatmap(matrix, width: int = 48, height: int = 24, title: str = "") -> str:
    """把任意 2D 数组渲染成字符热力图。

    用途：在没有图形界面（服务器 / Docker / SSH）时，依然能"看到"注意力权重、
    卷积特征图、位置编码矩阵等 2D 结构。
    """
    import numpy as np

    a = np.asarray(matrix, dtype=float)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.ndim > 2:
        a = a.reshape(a.shape[0], -1)

    h, w = a.shape
    # 最近邻下采样到目标尺寸
    row_idx = np.linspace(0, h - 1, height).astype(int)
    col_idx = np.linspace(0, w - 1, width).astype(int)
    small = a[row_idx][:, col_idx]

    lo, hi = float(small.min()), float(small.max())
    rng = hi - lo if hi > lo else 1.0
    norm = (small - lo) / rng
    lines = []
    if title:
        lines.append(title)
    for r in range(small.shape[0]):
        row = "".join(_ASCII_RAMP[min(int(v * (len(_ASCII_RAMP) - 1)), len(_ASCII_RAMP) - 1)] for v in norm[r])
        lines.append("|" + row + "|")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# 数值校验
# --------------------------------------------------------------------------------------


def check_close(a, b, tol: float = 1e-5, name: str = "tensor") -> bool:
    """对比两个数组/张量是否接近，返回 bool 并打印结论。常用于"手写实现 vs 框架实现"。"""
    import numpy as np

    def _to_numpy(x):
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=float)

    a_np, b_np = _to_numpy(a), _to_numpy(b)
    if a_np.shape != b_np.shape:
        print(f"[FAIL] {name}: 形状不一致 {a_np.shape} vs {b_np.shape}")
        return False
    max_err = float(np.max(np.abs(a_np - b_np)))
    ok = max_err <= tol
    print(f"[{'OK' if ok else 'FAIL'}] {name}: 最大绝对误差 = {max_err:.3e} (tol={tol:g})")
    return ok


def grad_check(f, x, analytic_grad, eps: float = 1e-6, tol: float = 1e-5, name: str = "grad") -> bool:
    """有限差分梯度检查：验证手写反向传播是否正确。

    f: 标量函数，接收 numpy 数组 x
    analytic_grad: 与 x 同形的解析梯度
    """
    import numpy as np

    x = np.asarray(x, dtype=float)
    numeric = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        f_plus = float(f(x))
        x[idx] = old - eps
        f_minus = float(f(x))
        x[idx] = old
        numeric[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return check_close(numeric, analytic_grad, tol=tol, name=name)


# --------------------------------------------------------------------------------------
# 杂项：参数量、显存
# --------------------------------------------------------------------------------------


def human_num(n: float) -> str:
    """1234567 -> '1.23M'，方便打印参数量。"""
    for unit in ("", "K", "M", "B", "T"):
        if abs(n) < 1000.0:
            return f"{n:.2f}{unit}" if unit else f"{n:.0f}"
        n /= 1000.0
    return f"{n:.2f}T"


def bytes_str(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.2f}{unit}"
        n /= 1024.0
    return f"{n:.2f}PB"


def count_params(module) -> int:
    """统计 torch 模块参数量（没装 torch 时返回 0）。"""
    if not hasattr(module, "parameters"):
        return 0
    return sum(p.numel() for p in module.parameters())


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]
