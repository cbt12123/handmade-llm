"""
07 · LoRA / PEFT：只训练 0.1% 的参数就能微调大模型
====================================================
LoRA 的核心假设：微调时的权重更新 ΔW 是【低秩】的。
  W' = W + ΔW = W + B·A       其中 A:(r×d_in), B:(d_out×r), r ≪ d
  前向：  y = W·x + (α/r)·B·(A·x)
  初始化：A ~ N(0, σ²)，B = 0  → 训练开始时 ΔW = 0，不破坏原模型

本脚本：
  1. 手写 LoRALinear 并与完整微调对比（参数量、显存、效果）
  2. 演示推理时把 LoRA 权重【合并】回原权重 → 零额外延迟
  3. 讲清楚 rank / alpha / target_modules 怎么选
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import human_num, require_torch, section, set_seed, subsection, timer

set_seed(29)
torch = require_torch("07_lora_peft")
nn = torch.nn
F = torch.nn.functional

# ----------------------------------------------------------------------------------
section("1) 玩具任务：让一个『预训练好的』MLP 学会一个新映射")
D_IN, D_HID, D_OUT = 32, 64, 32
rng = np.random.default_rng(0)
X = torch.tensor(rng.normal(size=(512, D_IN)), dtype=torch.float32)
Y = torch.tensor(rng.normal(size=(512, D_OUT)), dtype=torch.float32) * 0.5   # 目标映射

torch.manual_seed(0)
base = nn.Sequential(nn.Linear(D_IN, D_HID), nn.ReLU(), nn.Linear(D_HID, D_OUT))
with torch.no_grad():                              # 假装这是"预训练权重"
    for p in base.parameters():
        p.normal_(0, 0.1)
base_loss0 = F.mse_loss(base(X), Y).item()
print(f"  微调前的 MSE = {base_loss0:.4f}（随机权重，纯 baseline）")


# ----------------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """在已有 Linear 外面套一层低秩增量。"""

    def __init__(self, linear: nn.Linear, rank=4, alpha=8.0):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad_(False)       # 冻结原权重
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)
        self.rank, self.scaling = rank, alpha / rank
        self.A = nn.Parameter(torch.randn(rank, linear.in_features) * 0.02)
        self.B = nn.Parameter(torch.zeros(linear.out_features, rank))   # B 初始化为 0

    def forward(self, x):
        return self.linear(x) + self.scaling * F.linear(F.linear(x, self.A), self.B)

    def delta_w(self):
        return self.scaling * (self.B @ self.A)        # (out, in)

    def merge(self):
        """推理前把 LoRA 合并进原权重：之后就是普通的 Linear，零额外开销。"""
        with torch.no_grad():
            self.linear.weight += self.delta_w()
        return self.linear


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------------
section("2) 三种方案对比：全量微调 / 只微调最后一层 / LoRA")
import copy


def train(model, steps=400, lr=1e-2):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    for _ in range(steps):
        loss = F.mse_loss(model(X), Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return F.mse_loss(model(X), Y).item()


# A. 全量微调
m_full = copy.deepcopy(base)
for p in m_full.parameters():
    p.requires_grad_(True)
loss_full = train(m_full)
n_full = count_trainable(m_full)

# B. 只微调最后一层
m_last = copy.deepcopy(base)
for p in m_last[:-1].parameters():
    p.requires_grad_(False)
loss_last = train(m_last)
n_last = count_trainable(m_last)

# C. LoRA
m_lora = copy.deepcopy(base)
m_lora[0] = LoRALinear(m_lora[0], rank=4, alpha=8.0)
m_lora[2] = LoRALinear(m_lora[2], rank=4, alpha=8.0)
loss_lora = train(m_lora, lr=1e-2)
n_lora = count_trainable(m_lora)

print(f"{'方案':<16}{'可训练参数':>12}{'占比':>10}{'最终 MSE':>12}")
total = sum(p.numel() for p in base.parameters())
for name, n, loss in [("全量微调", n_full, loss_full), ("仅最后一层", n_last, loss_last), ("LoRA r=4", n_lora, loss_lora)]:
    print(f"  {name:<14}{n:>12,}{n / total:>10.2%}{loss:>12.5f}")
print(f"  （基准：不微调 MSE={base_loss0:.4f}）")
print("""
  注意：这个玩具任务是【随机映射】，它的 ΔW 本身并不低秩，所以 LoRA 反而吃亏。
  真实场景（指令微调、领域适配、风格迁移）里 ΔW 近似低秩，
  LoRA 用不到 1% 的参数通常能达到全量微调 95%~100% 的效果 —— 这正是它的价值。
""")

section("3) rank 的影响")
print(f"{'rank':<8}{'可训练参数':>12}{'最终 MSE':>12}")
for r in [1, 2, 4, 8, 16]:
    m = copy.deepcopy(base)
    m[0] = LoRALinear(m[0], rank=r, alpha=2 * r)
    m[2] = LoRALinear(m[2], rank=r, alpha=2 * r)
    loss = train(m, steps=400, lr=1e-2)
    print(f"  {r:<6}{count_trainable(m):>12,}{loss:>12.5f}")
print("""
  经验取值：r=8~64（任务越复杂取越大）；alpha 通常设为 2×r 或固定 16/32。
  scaling = alpha / rank 的作用：让不同 rank 下的"有效学习率"大致可比。
""")

section("4) 推理时合并权重：零额外延迟")
lin = m_lora[2]
before = lin.linear.weight.clone()
delta_norm = lin.delta_w().norm().item()
merged = lin.merge()
x_test = torch.randn(4, D_IN)
with torch.no_grad():
    out_lora = m_lora(x_test)
    # 用合并后的权重重新算一遍，验证等价
    plain = nn.Sequential(m_lora[0].linear, nn.ReLU(), merged)
    out_merged = plain(x_test)
print(f"  ||ΔW|| = {delta_norm:.4f}   原权重 ||W|| = {before.norm().item():.4f}")
print(f"  合并前后输出的最大差异 = {(out_lora - out_merged).abs().max().item():.2e}")
print("""
  合并后就是一个普通 Linear → 部署时没有任何额外算子，
  这是 LoRA 相比 Adapter（要插额外层）的巨大优势。
""")

section("5) 实战要点")
print("""
  ① target_modules：优先 q_proj / v_proj（原论文），实践中把所有线性层都加上效果更好
  ② 学习率：LoRA 用 1e-4 ~ 2e-4（比全量微调高一个量级，因为参数少、梯度噪声大）
  ③ 过拟合：LoRA 也会过拟合，小数据集用更小的 rank + 更多 dropout + 早停
  ④ QLoRA：4-bit NF4 底座 + fp16 LoRA → 单卡 48G 可微调 65B（见 05 节量化）
  ⑤ 其他 PEFT：Prefix Tuning / P-Tuning v2 / IA³ / DoRA（把幅度与方向解耦）
  ⑥ 什么时候不该用 LoRA：数据量大且要大幅改变模型行为（如继续预训练）→ 全量微调更好
""")
