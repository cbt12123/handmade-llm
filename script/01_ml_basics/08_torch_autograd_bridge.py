"""
08 · 从 numpy 手写梯度 到 PyTorch autograd（通往深度学习框架的桥）
==================================================================
从第 2 章开始我们会用 PyTorch，但你完全不需要忘记前面手写的那套东西，
因为 PyTorch 只是在帮你做两件事：

  1. 自动记录计算图，反向时自动链式求导（autograd）
  2. 把 for 循环里的矩阵乘法换成高度优化的 GPU 算子（第 7 章你会自己写算子）

本脚本用同一个 2 层 MLP 做对照，证明"手写梯度 == autograd 梯度"。
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import check_close, require_torch, section, set_seed, subsection

set_seed(17)
torch = require_torch("08_torch_autograd_bridge")

# ----------------------------------------------------------------------------------
# 1) autograd 的最小例子
# ----------------------------------------------------------------------------------
section("1) autograd 三步：requires_grad → backward → grad")
x = torch.tensor([[2.0]], requires_grad=True)
w = torch.tensor([[3.0]], requires_grad=True)
z = w * x                      # z = 3*2 = 6
L = z**2                       # L = 36
L.backward()                   # 反向传播
print(f"  x.grad = {x.grad.item()}  (dL/dx = 2·w²·x = {2 * 9 * 2})")
print(f"  w.grad = {w.grad.item()}  (dL/dw = 2·w·x² = {2 * 3 * 4})")
print("  关键：只要 tensor 的 requires_grad=True，PyTorch 就会记录它参与的所有运算。")

# ----------------------------------------------------------------------------------
# 2) 同一个 MLP：numpy 手写梯度 vs torch autograd
# ----------------------------------------------------------------------------------
section("2) 对照实验：numpy 手写 vs torch autograd")
n, d_in, d_h, d_out = 32, 8, 16, 1
rng = np.random.default_rng(0)
X_np = rng.normal(size=(n, d_in))
y_np = rng.normal(size=(n, d_out))
W1_np = rng.normal(0, 0.1, (d_in, d_h))
W2_np = rng.normal(0, 0.1, (d_h, d_out))


def numpy_grads(W1, W2):
    h = np.maximum(0, X_np @ W1)                       # ReLU
    pred = h @ W2
    loss = float(np.mean((pred - y_np) ** 2))
    g = 2.0 * (pred - y_np) / n
    dW2 = h.T @ g
    dh = g @ W2.T
    dh[h <= 0] = 0
    dW1 = X_np.T @ dh
    return loss, dW1, dW2


loss_np, dW1_np, dW2_np = numpy_grads(W1_np, W2_np)

# 注意 dtype：numpy 默认 float64，而 torch 的模型参数默认 float32，混用会报 dtype 错误
F32 = torch.float32
X_t = torch.tensor(X_np, dtype=F32)
y_t = torch.tensor(y_np, dtype=F32)
W1_t = torch.tensor(W1_np, dtype=F32, requires_grad=True)
W2_t = torch.tensor(W2_np, dtype=F32, requires_grad=True)
h_t = torch.relu(X_t @ W1_t)
pred_t = h_t @ W2_t
loss_t = torch.mean((pred_t - y_t) ** 2)
loss_t.backward()

print(f"  numpy loss = {loss_np:.8f}")
print(f"  torch loss = {loss_t.item():.8f}")
check_close(dW1_np, W1_t.grad, tol=1e-6, name="dL/dW1 手写 vs autograd")
check_close(dW2_np, W2_t.grad, tol=1e-6, name="dL/dW2 手写 vs autograd")

# ----------------------------------------------------------------------------------
# 3) PyTorch 训练循环标准模板（后面所有章节都长这样）
# ----------------------------------------------------------------------------------
section("3) PyTorch 训练循环模板（务必背下来）")
model = torch.nn.Sequential(
    torch.nn.Linear(d_in, d_h),
    torch.nn.ReLU(),
    torch.nn.Linear(d_h, d_out),
)
opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
loss_fn = torch.nn.MSELoss()

for step in range(300):
    pred = model(X_t)                 # ① 前向
    loss = loss_fn(pred, y_t)         # ② 算损失
    opt.zero_grad()                   # ③ 清空上一步梯度（不清会累加！）
    loss.backward()                   # ④ 反向
    opt.step()                        # ⑤ 更新参数
    if step % 100 == 0 or step == 299:
        print(f"  step {step:>4}  loss={loss.item():.6f}")

print("\n模板五步：zero_grad → forward → loss → backward → step")

subsection("4) 常用 API 速查（后面章节会反复出现）")
print("""
  torch.tensor / torch.from_numpy       : 建张量（注意 torch 默认 float32）
  x.view / x.reshape / x.transpose      : 变换形状（Transformer 里到处都是）
  torch.einsum("bhd,bkd->bhk", q, k)    : 爱因斯坦求和，注意力最清晰的写法
  torch.nn.Linear(d_in, d_out)          : 就是第 2 章的 XW + b
  torch.nn.functional.scaled_dot_product_attention : PyTorch 自带的融合注意力
  model.eval() + torch.no_grad()        : 推理时关掉 dropout 并停止建图，省显存
""")

section("5) 小结：你其实已经会深度学习了")
print("""
  前面 7 个脚本已经覆盖了全部核心机制：
    参数 → 前向 → 损失 → 反向 → 优化器更新
  后面所有内容（CNN / Transformer / LLM）只是"模块"变了，循环一模一样。
  第 7 章你会往下沉一层：连 torch 里的算子都自己用 CUDA 写。
""")
