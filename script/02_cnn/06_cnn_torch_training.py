"""
06 · 用 PyTorch 训练一个 CNN，并"看"它学到了什么
==================================================
任务：判别 32×32 图像里的图形 —— 圆 / 方 / 三角（纯合成，无需下载数据集）
目的：
  1. 熟悉 PyTorch 训练 CNN 的标准写法（后面所有章节都是这个模板）
  2. 打印训练后的第一层卷积核，验证"CNN 第一层会学到边缘检测器"
  3. 对比 CNN 与同参数量 MLP 的效果，体会"归纳偏置"的价值
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import ascii_heatmap, has_matplotlib, human_num, require_torch, save_fig, section, set_seed, subsection

set_seed(10)
torch = require_torch("06_cnn_torch_training")
F = torch.nn.functional

SIZE, N_PER = 32, 600          # 600 张：450 训练 / 150 测试


def make_shapes(n_per=N_PER, size=SIZE, seed=0):
    """合成三类图形：圆、方、三角。"""
    rng = np.random.default_rng(seed)
    imgs, labels = [], []
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(n_per):
        cls = rng.integers(0, 3)
        cx, cy = rng.integers(9, size - 9, 2)
        r = rng.integers(6, 10)
        if cls == 0:                                            # 圆
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r**2
        elif cls == 1:                                          # 方
            mask = (np.abs(xx - cx) < r) & (np.abs(yy - cy) < r)
        else:                                                   # 三角
            mask = (yy > cy) & (np.abs(xx - cx) < (yy - cy) * 0.8) & (yy - cy < r * 1.5)
        img = mask.astype(np.float32)
        img += rng.normal(0, 0.15, (size, size))                # 噪声
        imgs.append(img)
        labels.append(cls)
    return (np.array(imgs)[:, None].astype(np.float32), np.array(labels))


X_np, y_np = make_shapes()
Xtr, ytr = torch.tensor(X_np[:450]), torch.tensor(y_np[:450])
Xte, yte = torch.tensor(X_np[450:]), torch.tensor(y_np[450:])
print(f"训练集 {Xtr.shape}  测试集 {Xte.shape}")


class SmallCNN(torch.nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 3, padding=1)      # 32×32 → 32×32
        self.conv2 = torch.nn.Conv2d(8, 16, 3, padding=1)     # 32×32 → 32×32
        self.pool = torch.nn.MaxPool2d(2)                     # → 16×16 → 8×8
        self.fc = torch.nn.Linear(16 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return self.fc(x.flatten(1))


class SmallMLP(torch.nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 32, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train(model, n_train, epochs=20, lr=3e-3, verbose=False):
    """用前 n_train 个样本训练，在固定测试集上评估。"""
    Xa, ya = Xtr[:n_train], ytr[:n_train]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        logits = model(Xa)
        loss = F.cross_entropy(logits, ya)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xte).argmax(1) == yte).float().mean().item()
        if verbose and (ep % 4 == 0 or ep == epochs - 1):
            print(f"    epoch {ep:>2}  loss={loss.item():.4f}  测试准确率={acc:.4f}")
    return acc


section("1) 训练 CNN（300 张训练图，20 个 epoch）")
cnn = SmallCNN()
print(f"  CNN 参数量 = {human_num(float(sum(p.numel() for p in cnn.parameters())))}")
acc_cnn = train(cnn, n_train=300, verbose=True)

section("2) CNN vs MLP：样本量越小，归纳偏置越值钱")
print(f"{'训练样本数':<12}{'CNN 测试准确率':>18}{'MLP 测试准确率':>18}")
rows = {}
for n in (60, 150, 300):
    a_cnn = train(SmallCNN(), n_train=n)
    a_mlp = train(SmallMLP(), n_train=n)
    rows[n] = (a_cnn, a_mlp)
    print(f"  {n:<10}{a_cnn:>18.4f}{a_mlp:>18.4f}")
print(f"\n  MLP 参数量 = {human_num(float(sum(p.numel() for p in SmallMLP().parameters())))}"
      f"（是 CNN 的 {sum(p.numel() for p in SmallMLP().parameters()) / sum(p.numel() for p in SmallCNN().parameters()):.0f} 倍）")
print("  结论：数据少时 CNN 明显更好 —— 它不需要从数据里重新学习『局部性』和『平移不变性』，")
print("        这两个先验是写死在结构里的。这也是视觉任务上 CNN 至今仍有生命力的原因。")

section("3) 可视化训练后的第一层卷积核（8 个 3×3）")
w = cnn.conv1.weight.detach().numpy()                 # (8,1,3,3)
for i in range(w.shape[0]):
    print(f"\n  核 #{i}:")
    for row in w[i, 0]:
        print("    " + " ".join(f"{v:+6.2f}" for v in row))
print(ascii_heatmap(w.reshape(8, 9), width=54, height=8, title="\n  8 个核展平后的权重分布（亮=正，暗=负）："))
print("  观察：很多核呈现出『中心正、四周负』或『一侧正一侧负』的模式，")
print("        这正是 Laplacian / Sobel 类的边缘检测器——网络自己学出来了。")

section("4) 特征图：把一张真实图像喂进去看激活")
with torch.no_grad():
    feat = F.relu(cnn.conv1(Xte[:1]))                 # (1,8,32,32)
print(f"  特征图形状 = {tuple(feat.shape)}")
print(ascii_heatmap(feat[0, 0].numpy(), width=40, height=14, title="  第 0 个通道的激活："))

if has_matplotlib():
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 8, figsize=(13, 3.6))
    for i in range(8):
        axes[0, i].imshow(w[i, 0], cmap="coolwarm", vmin=-w.max(), vmax=w.max())
        axes[0, i].axis("off")
        axes[1, i].imshow(feat[0, i].numpy(), cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("核")
    axes[1, 0].set_ylabel("激活")
    save_fig(fig, Path(__file__).parent / "outputs" / "06_cnn_filters.png")

section("5) 结论速记")
print("""
  · PyTorch 训练 CNN 与第 1 章 MLP 的循环完全一样，只是模块换成了 Conv2d/Pool
  · Conv2d 的参数量 = C_in × C_out × K × K + C_out，与输入分辨率无关
  · 可视化权重与特征图是理解"模型在看什么"的第一步；LLM 时代这演化为
    attention 可视化、logit lens、稀疏自编码器（SAE）等可解释性方法
""")
