"""
03 · KV Cache：LLM 推理最重要的一个优化
========================================
问题：自回归生成时，第 t 步的注意力需要用到 1..t 所有位置的 K 和 V。
如果每步都重算一遍，总复杂度是 O(T²) —— 生成越长越慢。

解法：把已经算过的 K/V 缓存起来（KV Cache），每步只算"新 token"的 Q/K/V：
    K_cache = concat(K_cache, K_new)      V_cache = concat(V_cache, V_new)
    attn    = softmax(Q_new · K_cacheᵀ / √d) · V_cache

本脚本：
  1. 手写带 / 不带 KV Cache 的推理，验证输出完全一致
  2. 实测加速比（CPU 上也能看到明显差距）
  3. 算一算 KV Cache 到底吃多少显存，引出 GQA / MQA / PagedAttention
"""
import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import bytes_str, check_close, require_torch, section, set_seed, subsection, timer

set_seed(13)
torch = require_torch("03_kv_cache")
nn = torch.nn
F = torch.nn.functional

D_MODEL, H, LAYERS, VOCAB, MAX_LEN = 64, 4, 2, 40, 256


class TinyLM(nn.Module):
    """极简 decoder-only 模型，注意力手写，方便插入/关闭 KV Cache。"""

    def __init__(self):
        super().__init__()
        self.h, self.dh = H, D_MODEL // H
        self.emb = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Embedding(MAX_LEN, D_MODEL)
        self.wq = nn.ModuleList([nn.Linear(D_MODEL, D_MODEL, bias=False) for _ in range(LAYERS)])
        self.wk = nn.ModuleList([nn.Linear(D_MODEL, D_MODEL, bias=False) for _ in range(LAYERS)])
        self.wv = nn.ModuleList([nn.Linear(D_MODEL, D_MODEL, bias=False) for _ in range(LAYERS)])
        self.wo = nn.ModuleList([nn.Linear(D_MODEL, D_MODEL, bias=False) for _ in range(LAYERS)])
        self.ff = nn.ModuleList([nn.Sequential(nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(),
                                               nn.Linear(4 * D_MODEL, D_MODEL)) for _ in range(LAYERS)])
        self.ln1 = nn.ModuleList([nn.LayerNorm(D_MODEL) for _ in range(LAYERS)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(D_MODEL) for _ in range(LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)

    def _attn(self, x, layer, k_cache=None, v_cache=None):
        """x: (B, T, D)。若传入 cache，则把新的 K/V 拼进去再算注意力。"""
        B, T, _ = x.shape
        q = self.wq[layer](x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.wk[layer](x).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.wv[layer](x).view(B, T, self.h, self.dh).transpose(1, 2)
        if k_cache is not None:
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        scores = (q @ k.transpose(-2, -1)) / (self.dh ** 0.5)
        Tk = k.shape[2]
        causal = torch.ones(T, Tk, dtype=torch.bool, device=x.device).tril(diagonal=Tk - T)
        scores = scores.masked_fill(~causal, float("-inf"))
        w = torch.softmax(scores, dim=-1)
        ctx = (w @ v).transpose(1, 2).reshape(B, T, D_MODEL)
        return self.wo[layer](ctx), k, v

    def forward(self, idx, use_cache=False, past=None):
        B, T = idx.shape
        x = self.emb(idx) + self.pos(torch.arange(T, device=idx.device)[None])
        new_past = [] if use_cache else None
        for l in range(LAYERS):
            h = self.ln1[l](x)
            kc = past[l][0] if (use_cache and past is not None) else None
            vc = past[l][1] if (use_cache and past is not None) else None
            o, k, v = self._attn(h, l, kc, vc)
            x = x + o
            x = x + self.ff[l](self.ln2[l](x))
            if use_cache:
                new_past.append((k, v))
        logits = self.head(self.ln_f(x))
        return (logits, new_past) if use_cache else logits


section("1) 构造一个随机初始化（未训练）的小模型，只比较计算方式")
model = TinyLM()
model.eval()
print(f"  参数量 = {sum(p.numel() for p in model.parameters()):,}")

prefix = torch.randint(0, VOCAB, (1, 8))
GEN = 64

# ----------------------------------------------------------------------------------
section("2) 无 KV Cache：每一步都把整个序列重算一遍")


def generate_no_cache(model, prefix, n_new):
    ctx = prefix.clone()
    outs = []
    for _ in range(n_new):
        logits = model(ctx)                    # ← 整段重算
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        outs.append(int(nxt.item()))
        ctx = torch.cat([ctx, nxt], dim=1)
    return outs, ctx


t0 = time.perf_counter()
out_nocache, ctx_nocache = generate_no_cache(model, prefix, GEN)
t_nocache = time.perf_counter() - t0

section("3) 有 KV Cache：每步只算新 token")


def generate_with_cache(model, prefix, n_new):
    ctx = prefix.clone()
    outs = []
    past = None
    logits, past = model(ctx, use_cache=True, past=None)      # prefill：一次算完整段
    nxt = logits[:, -1].argmax(-1, keepdim=True)
    outs.append(int(nxt.item()))
    pos = ctx.shape[1]
    for _ in range(n_new - 1):
        # decode：只喂 1 个 token，位置编号要接着走
        step_pos = torch.full((1, 1), pos, device=ctx.device)
        emb = model.emb(nxt) + model.pos(step_pos)
        new_past = []
        x = emb
        for l in range(LAYERS):
            h = model.ln1[l](x)
            o, k, v = model._attn(h, l, past[l][0], past[l][1])
            x = x + o
            x = x + model.ff[l](model.ln2[l](x))
            new_past.append((k, v))
        past = new_past
        pos += 1
        logits = model.head(model.ln_f(x))
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        outs.append(int(nxt.item()))
    return outs


t0 = time.perf_counter()
out_cache = generate_with_cache(model, prefix, GEN)
t_cache = time.perf_counter() - t0

print(f"  无 cache: {t_nocache * 1000:8.1f} ms")
print(f"  有 cache: {t_cache * 1000:8.1f} ms")
print(f"  加速比  : {t_nocache / t_cache:.2f}×   （生成 {GEN} 个 token，前缀 {prefix.shape[1]}）")
print(f"\n  两者生成的 token 完全一致: {out_nocache == out_cache}")
check_close(np.array(out_nocache), np.array(out_cache), tol=0, name="KV Cache 输出 vs 全量重算输出")
print("  注意：KV Cache 是【精确等价】的优化，不改变任何数值结果（不同于量化/剪枝）。")

section("4) KV Cache 的代价：显存")
for seq in [1024, 4096, 16384, 65536]:
    mem = 2 * LAYERS * seq * H * (D_MODEL // H) * 4       # fp32，K 和 V 两份
    print(f"  本玩具模型 seq={seq:>6}: {bytes_str(mem)}")
print()
for name, (layers, kv_heads, hd) in [("Llama-3-8B (GQA 8 KV头)", (32, 8, 128)),
                                     ("Llama-3-8B (若为 MHA 32头)", (32, 32, 128)),
                                     ("Llama-3-70B (GQA 8 KV头)", (80, 8, 128))]:
    for seq in [4096, 32768, 131072]:
        mem = 2 * layers * seq * kv_heads * hd * 2        # fp16
        print(f"  {name:<28} seq={seq:>7}: {bytes_str(mem):>10}")
print("""
  结论：KV Cache 与"层数 × KV头数 × head_dim × 序列长度 × batch"成正比
  三条优化主线（第 6 章展开）：
    ① GQA / MQA：减少 KV 头数 → 显存直接降到 1/4 ~ 1/32
    ② MLA（DeepSeek）：把 KV 压成一个低秩潜向量
    ③ PagedAttention（vLLM）：把 KV Cache 分页管理，消除碎片（下面第 5 节模拟）
""")

section("5) Prefill / Decode 两阶段 + 为什么需要 PagedAttention")


def prefill_decode_flops(T_prompt, T_gen, d=D_MODEL):
    """粗略估算：prefill 是 T² 级，decode 每步是 T 级"""
    prefill = 2 * d * T_prompt**2
    decode = 2 * d * sum(T_prompt + i for i in range(T_gen))
    return prefill, decode


for tp, tg in [(512, 128), (2048, 512), (8192, 1024)]:
    p, d = prefill_decode_flops(tp, tg)
    print(f"  prompt={tp:>5}, 生成={tg:>5} → prefill FLOPs≈{p / 1e6:8.1f}M, "
          f"decode FLOPs≈{d / 1e6:8.1f}M, decode 占比={d / (p + d):.1%}")
print("""
  长 prompt + 短输出 → prefill 主导（拼算力）
  短 prompt + 长输出 → decode 主导（拼显存带宽）
  PagedAttention 解决的是"显存碎片"：预分配最大长度会浪费 60%+ 显存，
  分页后按需分配，同样的卡能塞下 2~4 倍的并发请求。
""")
