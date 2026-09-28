"""第 7 章（NVIDIA 算子）的公共工具：环境探测、nvcc 编译、优雅降级。

设计目标：**没有 GPU 的机器也能跑完本章所有脚本**（会打印 [SKIP] 并给出 CPU 版模拟结果），
在有 NVIDIA 显卡 + nvcc 的机器上则真正编译并跑 CUDA kernel。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent))            # 让 common_utils 可 import


def has_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def nvcc_available() -> bool:
    return shutil.which("nvcc") is not None


def has_triton() -> bool:
    try:
        import triton  # noqa: F401

        return True
    except Exception:
        return False


def skip(msg: str):
    """打印 [SKIP] 并正常退出（run_all.sh 会把退出码 0 且带 [SKIP] 的脚本标记为跳过）。"""
    print(f"[SKIP] {msg}")
    raise SystemExit(0)


def print_env():
    """打印一屏环境信息：GPU 型号、算力、显存、nvcc、torch 版本。"""
    print("  Python     :", sys.version.split()[0])
    if not has_torch():
        print("  torch      : 未安装（pip install torch）")
        print("  nvidia-smi :", "无" if shutil.which("nvidia-smi") is None else "有驱动")
        return
    import torch

    print("  torch      :", torch.__version__, "| CUDA 版本:", torch.version.cuda)
    print("  GPU 可用   :", torch.cuda.is_available())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"    [{i}] {p.name}  SM 数量={p.multi_processor_count}  "
                  f"算力=sm_{p.major}{p.minor}  显存={p.total_memory / 1024**3:.1f}GB  "
                  f"带宽≈{p.memory_clock_rate * p.memory_bus_width / 8 / 1e6:.0f}GB/s(理论)")
    print("  nvcc       :", shutil.which("nvcc") or "未找到（需要 CUDA Toolkit 的 devel 镜像）")
    print("  TORCH_CUDA_ARCH_LIST =", os.environ.get("TORCH_CUDA_ARCH_LIST", "(未设置，将自动探测)"))


def arch_flags() -> list[str]:
    """生成 -gencode 参数；优先读环境变量，否则按当前 GPU 的算力自动填。"""
    env = os.environ.get("TORCH_CUDA_ARCH_LIST")
    codes: list[str] = []
    if env:
        for a in env.replace(";", " ").split():
            a = a.strip()
            if not a:
                continue
            a = a.replace(".", "")
            if a.startswith("sm_") or a.isdigit():
                num = a[3:] if a.startswith("sm_") else a
                codes += ["-gencode", f"arch=compute_{num},code=sm_{num}"]
    elif cuda_available():
        import torch

        p = torch.cuda.get_device_properties(0)
        num = f"{p.major}{p.minor}"
        codes += ["-gencode", f"arch=compute_{num},code=sm_{num}"]
    else:
        codes += ["-gencode", "arch=compute_80,code=sm_80"]      # 兜底：A100 常见算力
    return codes


def compile_cu(src: Path, out: Path, extra_flags=None, libs=None, verbose=False) -> Path:
    """用 nvcc 编译一个 .cu 文件，返回可执行文件/目标文件路径。"""
    if not nvcc_available():
        skip("未找到 nvcc：请在 nvidia/cuda-devel 镜像中运行（docker/Dockerfile），或 apt install nvidia-cuda-toolkit")
    cmd = ["nvcc", str(src), "-o", str(out), "-O3", "--std=c++17"] + arch_flags()
    cmd += extra_flags or []
    cmd += [f"-l{lib}" for lib in (libs or [])]
    print("  $", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-3000:])
        raise RuntimeError("nvcc 编译失败（常见原因：算力不匹配，请设置 TORCH_CUDA_ARCH_LIST）")
    if verbose:
        print(r.stderr[-800:])
    return out


def run_binary(path: Path) -> str:
    r = subprocess.run([str(path)], capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        print(out[-2000:])
        raise RuntimeError("运行失败")
    return out


def ensure_dir(name="build") -> Path:
    d = HERE / name
    d.mkdir(exist_ok=True)
    return d
