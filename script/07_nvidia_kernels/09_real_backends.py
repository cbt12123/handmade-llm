"""
09 · 真实生态：让真实模型跑在更快的 kernel 上
==================================================
第 7 章前面手写了 kernel；真实项目中你更常做的是"选对后端"：

  transformers 的 attn_implementation 参数
    eager               : 朴素实现（会真的构造 T×T 矩阵），支持输出注意力权重
    sdpa                : PyTorch 的 scaled_dot_product_attention（默认）
    flash_attention_2   : FlashAttention-2（需 Ampere+ GPU + pip install flash-attn）

本节：
  1. 检测当前环境支持哪些后端
  2. 用真实模型对比 eager / sdpa 的速度
  3. torch.compile 一行加速
  4. 在 GPU 上启用 flash-attn 与 fp8/bf16 的完整配置
"""
import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from _cuda_helpers import cuda_available
from common_utils import has_module, require_module, section, set_seed, subsection, timer

set_seed(0)
tf = require_module("transformers", "第 7 章前 8 节不依赖它", pip_name="transformers")
torch = require_module("torch", "需要 torch", pip_name="torch")
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------------------------------------------------------------
section("1) 当前环境支持哪些注意力后端")
print(f"  torch 版本          : {torch.__version__}")
print(f"  CUDA 可用           : {cuda_available()}")
print(f"  flash-attn 已安装    : {has_module('flash_attn')}")
if cuda_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU                 : {p.name} (sm_{p.major}{p.minor})")
    print(f"  flash-attn 可用     : {has_module('flash_attn') and p.major >= 8}")
else:
    print("  flash-attn 可用     : 否（需要 NVIDIA GPU，算力 ≥ 8.0）")

print("""
  后端选择规则（transformers 内部也是这么判断的）：
    · 没 GPU 或没装 flash-attn → sdpa（或 eager）
    · 有 GPU + flash-attn + bf16/fp16 → flash_attention_2（最快，省显存）
    · 需要 output_attentions=True（做可视化）→ 只能用 eager
    · 训练时想用最稳的 → sdpa（flash-attn 对变长/padding 支持更挑）
""")

# ----------------------------------------------------------------------------------
section("2) 加载真实模型并对比后端")
MODEL = "hf-internal-testing/tiny-random-gpt2"
try:
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager")
    base.eval()
except Exception as e:
    print(f"[SKIP] 无法加载模型（{type(e).__name__}）：本节需要联网")
    raise SystemExit(0)

ids = torch.randint(0, 900, (2, 64))          # batch=2, 序列 64

def bench_model(model, iters=20):
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            model(ids)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(ids)
    return (time.perf_counter() - t0) / iters * 1000

t_eager = bench_model(base)
print(f"  eager : {t_eager:8.3f} ms/次")

try:
    sdpa_model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="sdpa")
    t_sdpa = bench_model(sdpa_model)
    print(f"  sdpa  : {t_sdpa:8.3f} ms/次    加速比 {t_eager / t_sdpa:.2f}×")
except Exception as e:
    print(f"  sdpa  : 不可用（{type(e).__name__}）")

print("""
  在 tiny 模型 + CPU 上差距不大（计算本身太小）；
  真实 7B 模型 + 长序列时，flash/sdpa 相比 eager 通常是数倍差距，且显存从 O(T²) 降到 O(T)。
""")

# ----------------------------------------------------------------------------------
section("3) torch.compile：一行换加速")
print("""
```python
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", dtype=torch.bfloat16).cuda()
compiled = torch.compile(model)                 # 默认 mode
compiled = torch.compile(model, mode="reduce-overhead")   # 用 CUDA Graph 降低启动开销
compiled = torch.compile(model, mode="max-autotune")      # 启动时自动调优（编译慢，运行快）
```

  原理：TorchDynamo 抓图 → AOTAutograd → Inductor 生成融合的 Triton kernel
  收益：把逐元素算子（GELU、LayerNorm、残差）融合掉，减少访存往返
  注意：
    · 第一次调用会编译（几十秒），之后才快
    · 输入形状频繁变化会导致反复重编译 → 用 dynamic=False 或 padding 到固定形状
""")
try:
    import torch._dynamo as dynamo

    compiled = torch.compile(base, backend="aot_eager")     # CPU 安全后端
    with torch.no_grad():
        _ = compiled(ids[:, :16])                            # 触发编译
    print(f"  torch.compile 在本机可用（backend=aot_eager）→ 已跑通一次前向")
except Exception as e:
    print(f"  [SKIP] torch.compile 在本机不可用：{type(e).__name__}")

# ----------------------------------------------------------------------------------
section("4) GPU 上的完整加速配置（照抄即可）")
print("""
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    dtype=torch.bfloat16,                       # A100/H100 用 bf16；老卡用 fp16
    device_map="auto",                          # 多卡自动切分（accelerate）
    attn_implementation="flash_attention_2",    # 需要先 pip install flash-attn
)
model.config.use_cache = True                   # 推理务必打开 KV Cache
```
安装 flash-attn（编译较慢，建议直接用预编译 wheel）：
```bash
pip install flash-attn --no-build-isolation
# 或（推荐）直接用带好的镜像：
docker run --gpus all -it llm-line:cu121 bash
```
验证它真的在用 flash：
```python
import flash_attn; print(flash_attn.__version__)
# 或者看 nsys/ncu 里有没有 flash_fwd_kernel
```
""")

section("5) 各类加速手段的收益排序（经验）")
print("""
  收益从大到小（针对 LLM 推理）：
    ① 用对框架：vLLM / TGI（连续批处理 + PagedAttention）     3~10×
    ② KV Cache                                                5~20×（越长越明显）
    ③ 量化（W4A16 / fp8）                                     1.5~3×（decode 阶段）
    ④ FlashAttention                                          1.5~2× + 省 O(T²) 显存
    ⑤ torch.compile / CUDA Graph                              1.1~1.5×（小 batch 更明显）
    ⑥ 手写融合算子（RMSNorm+Residual 等）                      5%~20%
  结论：先把①②③做对，再考虑写算子。算子是"最后一公里"的优化。
""")
