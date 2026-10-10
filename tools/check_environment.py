"""Read-only environment checks; run with the Python used for the lesson."""
import argparse
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROFILES = {
    "base": ["numpy", "matplotlib", "requests"],
    "ml": ["numpy", "matplotlib", "sklearn", "torch"],
    "llm": ["numpy", "requests", "torch", "transformers", "fastapi", "uvicorn"],
    "gpu": ["numpy", "torch", "triton", "transformers"],
    "cuda-dev": [],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--model-path", type=Path)
    args = parser.parse_args()
    checks = []

    def record(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})

    record("python", sys.version_info >= (3, 10), sys.version.split()[0])
    modules = {}
    for name in PROFILES[args.profile]:
        try:
            modules[name] = importlib.import_module(name)
            record(name, True, getattr(modules[name], "__version__", "import OK"))
        except Exception as exc:
            record(name, False, f"{type(exc).__name__}: {exc}")
    if args.profile == "gpu":
        try:
            torch = modules["torch"]
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable in this Python environment")
            x = torch.ones((32, 32), device="cuda")
            y = x @ x
            torch.cuda.synchronize()
            if not bool(torch.all(y == 32).item()):
                raise RuntimeError("CUDA matrix result is incorrect")
            record("cuda_compute", True, torch.cuda.get_device_name(0))
        except Exception as exc:
            record("cuda_compute", False, f"{type(exc).__name__}: {exc}")
    if args.profile == "cuda-dev":
        for command in ("nvcc", "g++", "compute-sanitizer"):
            executable = shutil.which(command)
            if executable is None:
                record(command, False, "executable not found on PATH")
                continue
            try:
                result = subprocess.run([executable, "--version"], capture_output=True,
                                        text=True, timeout=20, check=True)
                record(command, True, (result.stdout + result.stderr).strip())
            except Exception as exc:
                record(command, False, f"{type(exc).__name__}: {exc}")
        record("cublas_header", Path("/usr/local/cuda/include/cublas_v2.h").is_file(),
               "/usr/local/cuda/include/cublas_v2.h")
    if args.model_path is not None:
        path = args.model_path
        record("model_config", (path / "config.json").is_file(), path / "config.json")
        record("model_weights", any(path.glob("*.safetensors")), path)
        record("model_tokenizer", (path / "tokenizer_config.json").is_file()
               and ((path / "tokenizer.json").is_file() or (path / "vocab.json").is_file()), path)
    passed = all(check["ok"] for check in checks)
    print(json.dumps({"profile": args.profile, "python_executable": sys.executable,
                      "passed": passed, "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
