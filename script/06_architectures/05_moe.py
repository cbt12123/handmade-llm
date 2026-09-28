"""
05 · MoE（Mixture of Experts）：用稀疏激活把模型"变大"而不变慢
================================================================
核心思想：把 FFN 换成 N 个"专家 FFN"，每个 token 只走其中 k 个（k=1 或 2）：

    y = Σ_{i ∈ TopK(g(x))} g_i(x) · E_i(x)

  · 总参数 = N × 单专家参数   → 模型"看起来"很大
  · 激活参数 = k × 单专家参数 → 每步的计算量与单专家模型同量级

代表模型：Switch Transformer（k=1）、Mixtral 8×7B（8 选 2）、DeepSeek-V3（256 选 8 + 共享专家）

本脚本：
  1. 手写 MoE 层（路由 + 分发 + 合并）
  2. 实现负载均衡损失，用实验证明"不加损失 → 专家塌缩"
  3. 训练一个玩具 MoE，观察【专家自发分化】
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import human_num, require_torch, section, set_seed, subsection, timer

set_seed(11)
torch = require_torch("05_moe")
nn = torch.nn
F = torch.nn.functional


class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=4, top_k=2, capacity_factor=1.25):
        super().__init__()
        self.num_experts, self.top_k = num_experts, top_k
        self.capacity_factor = capacity_factor
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x, return_aux=False):
        """x: (B, T, d) → y 同形"""
        B, T, d = x.shape
        xf = x.reshape(-1, d)                                        # (N, d)
        logits = self.gate(xf)                                       # (N, E)
        probs = torch.softmax(logits, dim=-1)
        topk_p, topk_idx = torch.topk(probs, self.top_k, dim=-1)     # (N, k)
        topk_p = topk_p / topk_p.sum(-1, keepdim=True)

        y = torch.zeros_like(xf)
        # —— 朴素实现：按专家分组处理（真实实现用 cumsum + 稀疏矩阵，见文末说明）——
        for e in range(self.num_experts):
            mask = (topk_idx == e).any(dim=-1)                       # 选中该专家的 token
            if not mask.any():
                continue
            weight = (topk_idx == e) * topk_p                        # (N, k) → 取对应权重
            w = weight.sum(-1, keepdim=True)[mask]
            y[mask] += w * self.experts[e](xf[mask])

        if not return_aux:
            return y.reshape(B, T, d)

        # ---- Switch Transformer 的负载均衡损失 ----
        # f_i: 实际被分到专家 i 的 token 比例；P_i: 路由概率的平均
        one_hot = F.one_hot(topk_idx, self.num_experts).float()      # (N, k, E)
        f = one_hot.sum(dim=(0, 1)) / (xf.shape[0] * self.top_k)
        P = probs.mean(dim=0)
        aux_loss = self.num_experts * (f * P).sum()
        return y.reshape(B, T, d), aux_loss, f


section("1) 参数量：MoE 的『看起来很大』")
d_model, d_ff, E, k = 512, 2048, 8, 2
dense_ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
moe = MoELayer(d_model, d_ff, num_experts=E, top_k=k)
p_dense = sum(p.numel() for p in dense_ffn.parameters())
p_moe = sum(p.numel() for p in moe.parameters())
print(f"  普通 FFN 参数    = {human_num(float(p_dense))}")
print(f"  MoE({E} 专家) 参数 = {human_num(float(p_moe))}   （×{p_moe / p_dense:.1f}）")
print(f"  激活参数（top-{k}）= {human_num(float(2 * p_dense * k / 1))} 的 FFN 部分 "
      f"→ 每 token 只算 {k}/{E} = {k / E:.0%} 的专家")
print("  这就是 Mixtral-8x7B 名字的由来：8 个 7B 专家，激活 2 个 → 47B 总参数 / 13B 激活参数。")

section("2) 专家塌缩（routing collapse）与负载均衡损失")
print("""
  问题：如果路由只把 token 送给少数几个专家 → 其余专家学不到东西 → 越学越偏（马太效应）
  Switch Transformer 的解决：加一个辅助损失
      L_aux = E · Σ_i f_i · P_i
      f_i = 实际分配比例（不可导，用 one-hot 统计）
      P_i = 路由概率均值（可导）
  理想均匀分布时 f_i = P_i = 1/E → L_aux = E · E · (1/E²) = 1（最小值）
""")
x_demo = torch.randn(64, 16, d_model)
_, aux, f = moe(x_demo, return_aux=True)
print(f"  随机初始化时各专家的负载比例: {np.round(f.detach().numpy(), 3)}")
print(f"  负载均衡损失 = {aux.item():.4f}（越接近 1 越均衡）")

section("3) 训练一个玩具 MoE：专家会自发分化吗？")
# 造数据：同一个输入空间里有【两种截然不同的映射】，由输入本身决定走哪条规则
rng = np.random.default_rng(0)
N = 4000
X_np = rng.normal(size=(N, 32)).astype(np.float32)
label = (rng.random(N) < 0.5).astype(np.int64)          # 隐式的"领域标签"
# 第 0 维是"领域标志位"：+1 = 领域 A，−1 = 领域 B（路由只看这一维就能分开）
X_np[:, 0] = np.where(label == 0, 1.0, -1.0) + rng.normal(0, 0.05, N)
X_np[label == 0, 16:] *= 0.05                            # 领域 A：只有前半维携带信息
X_np[label == 1, 1:16] *= 0.05                           # 领域 B：只有后半维携带信息
y_np = np.where(label == 0,
                np.sin(3 * X_np[:, 1:16].sum(1)),        # 规则 A：正弦
                np.abs(X_np[:, 16:].sum(1)) - 1.0)       # 规则 B：绝对值
Xt = torch.tensor(X_np)
yt = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)

torch.manual_seed(0)
toy_moe = MoELayer(32, 64, num_experts=4, top_k=1)
readout = nn.Linear(32, 1)
opt = torch.optim.AdamW(list(toy_moe.parameters()) + list(readout.parameters()), lr=3e-3)
AUX_W = 0.002                                            # 辅助损失权重：太小会塌缩，太大就均匀
with timer("训练耗时"):
    for step in range(600):
        h, aux, f = toy_moe(Xt[:, None], return_aux=True)
        pred = readout(h[:, 0])
        loss = F.mse_loss(pred, yt) + AUX_W * aux
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 300 == 0 or step == 599:
            print(f"  step {step:>4}  mse={F.mse_loss(pred, yt).item():.5f}  "
                  f"aux={aux.item():.4f}  负载={np.round(f.detach().numpy(), 2)}")

with torch.no_grad():
    _, top1 = torch.topk(torch.softmax(toy_moe.gate(Xt), -1), 1, dim=-1)
    assign = top1.squeeze()
    for c, name in [(0, "领域 A（正弦）"), (1, "领域 B（绝对值）")]:
        cnt = np.bincount(assign[np.array(label) == c].numpy(), minlength=4)
        print(f"  {name:<16} 的 token 被分配到的专家分布: {np.round(cnt / cnt.sum(), 3)}")
print("""
  如果两个领域各自"垄断"了不同的专家，就说明路由学到了有意义的划分 —— 这就是 MoE 的初衷：
  用不同的专家装不同的知识，而不是让一个稠密 FFN 硬记所有模式。
  （辅助损失权重越小，分化越明显；调到 0 就可能出现"只用一个专家"的塌缩。）
""")

section("4) 工程现实：MoE 的代价")
print("""
  ① 显存：参数还是要全部装进显存（47B 的 Mixtral 需要 ~90GB fp16），
     但可以用"专家并行"（EP）把不同专家放到不同卡上
  ② 通信：token 要被送到专家所在的卡 → all-to-all 通信是 MoE 的主要开销
  ③ 负载均衡：除辅助损失外，还有 capacity factor（每个专家最多接多少 token），
     超容量的 token 会被"丢弃"（直接走残差）→ 影响效果
  ④ 微调更易过拟合：MoE 参数多、每个专家数据少 → 常用更大的 dropout / 更多数据
  ⑤ DeepSeek 的创新：
     · 细粒度专家（把专家切得更小、数量更多）+ 共享专家（所有 token 都过）
     · 无辅助损失负载均衡：给每个专家加一个可动态调整的 bias，根据负载增减
       → 避免辅助损失对模型质量的干扰
""")

section("5) 什么时候该用 MoE")
print("""
  适合：服务吞吐优先、预训练算力充足、模型规模 > 30B
  不适合：显存极度受限（端侧）、微调数据很少、要求极致低延迟
  经验：同样"激活参数"下，MoE 通常比 dense 模型效果更好（更多参数 = 更多知识容量），
       但显存与工程复杂度显著上升。
""")
