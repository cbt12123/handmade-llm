"""Paths, transparent references, CUDA requirements, and timing helpers."""
from pathlib import Path
import json
import os
import statistics
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs' / 'part06'


def save(name, value):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('真实GPU实验需要CUDA；CPU概念实验不能当作GPU验证结果')
    torch.manual_seed(42)
    return torch


def rms_reference(x,weight,eps=1e-6):
    """Match Qwen's FP32 variance and normalization, then cast before weight."""
    dtype = x.dtype
    hidden = x.float()
    variance = hidden.square().mean(-1,keepdim=True)
    normalized = hidden * (variance+eps).rsqrt()
    return normalized.to(dtype) * weight


def bench(fn, iterations=80, repeats=7, warmup=20):
    import torch
    for _ in range(warmup):
        result = fn()
    torch.cuda.synchronize()
    event_samples, wall_samples = [], []
    for _ in range(repeats):
        start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        clock=time.perf_counter();start.record()
        for _ in range(iterations):
            result=fn()
        end.record();end.synchronize()
        wall_samples.append((time.perf_counter()-clock)*1000/iterations)
        event_samples.append(start.elapsed_time(end)/iterations)
    return {'iterations_per_repeat':iterations,'repeats':repeats,
            'event_median_ms':statistics.median(event_samples),'event_samples_ms':event_samples,
            'wall_median_ms':statistics.median(wall_samples),'wall_samples_ms':wall_samples,
            'scope':'Repeated eager calls; CUDA event window includes GPU idle gaps caused by host dispatch.'}


def model_path():
    path=Path(os.environ.get('LLM_MODEL_PATH','/models/qwen'))
    if not (path/'config.json').is_file():
        raise FileNotFoundError('设置 LLM_MODEL_PATH 为已有完整 Qwen 模型目录')
    return path
