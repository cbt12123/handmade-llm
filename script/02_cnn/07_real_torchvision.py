"""
07 · 真实生态：torchvision 里的 ResNet + 真实预训练权重
==========================================================
第 2 章我们手写了 conv2d；真实项目里没人手写 ResNet —— 直接 `torchvision.models` 拿。

本节：
  1. 加载真实的 ResNet-18，打印结构与参数量（和手写 LeNet 做对比）
  2. 逐层打印特征图尺寸变化（理解 stride=2 的下采样发生在哪）
  3. 加载 ImageNet 预训练权重，对真实图片做 top-5 分类
  4. torchvision.transforms 的标准预处理管线

需要联网下载权重（约 45MB）；没有网络时会 [SKIP]，其余部分照常运行。
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import human_num, require_module, require_torch, section, set_seed, subsection, timer

set_seed(0)
torch = require_torch("07_real_torchvision")
nn = torch.nn
F = torch.nn.functional
tv = require_module("torchvision", "其余章节不依赖它", pip_name="torchvision")
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

# ----------------------------------------------------------------------------------
section("1) 真实的 ResNet-18：结构 + 参数量")
model = models.resnet18(weights=None)
total = sum(p.numel() for p in model.parameters())
print(f"  ResNet-18 参数量 = {human_num(float(total))}")
print(f"  第 2 章手写的 LeNet(缩小版) ≈ 10.86K   →  相差 {total / 10860:.0f} 倍")

print("\n  卷积层的通道数变化（注意每隔几个 stage 就翻倍，同时分辨率减半）：")
print(f"  {'层':<12}{'权重形状':<28}{'输出分辨率(224输入)':<20}")
res = 224
for name in ["conv1", "layer1.0.conv1", "layer2.0.conv1", "layer3.0.conv1", "layer4.0.conv1"]:
    w = dict(model.named_parameters())[name + ".weight"]
    print(f"  {name:<12}{str(tuple(w.shape)):<28}{int(res):>6}×{int(res)}")
    res /= 2                                    # 每个 stage 的第一个 conv 用 stride=2
print(f"  最后接 AdaptiveAvgPool → 512 维 → fc(512, 1000)")

section("2) 逐层跑一遍，看特征图尺寸怎么变小")
x = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    h = model.conv1(x);           print(f"  conv1   → {tuple(h.shape)}")
    h = model.bn1(h); h = model.relu(h)
    h = model.maxpool(h);         print(f"  maxpool → {tuple(h.shape)}")
    for i, layer in enumerate([model.layer1, model.layer2, model.layer3, model.layer4], 1):
        h = layer(h)
        print(f"  layer{i}  → {tuple(h.shape)}")
    h = model.avgpool(h);         print(f"  avgpool → {tuple(h.shape)}")
    logits = model.fc(h.flatten(1))
print(f"  fc      → {tuple(logits.shape)}  （ImageNet 1000 类）")

section("3) torchvision.transforms：真实项目里的标准预处理")
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],      # ImageNet 统计值，必须一致
                         std=[0.229, 0.224, 0.225]),
])
print("""
  训练时的增强管线通常再加：RandomResizedCrop / RandomHorizontalFlip / ColorJitter / RandAugment
  ⚠️ 归一化参数必须与预训练时一致，否则精度会掉十几个点 —— 这是最常见的部署 bug。
""")

section("4) 加载 ImageNet 预训练权重做真实分类")
try:
    with timer("下载并加载预训练权重"):
        weights = ResNet18_Weights.DEFAULT
        net = models.resnet18(weights=weights)
        net.eval()
    categories = weights.meta["categories"]
    print(f"  类别数 = {len(categories)}")
except Exception as e:
    print(f"[SKIP] 无法下载预训练权重（{type(e).__name__}）：离线环境下可跳过本小节")
    net, categories = None, None

if net is not None:
    # 构造一张"有结构"的合成图（虽然不是真实照片，但能看到管线跑通）
    # 更真实的做法：用一张真实图片（见文末命令）
    rng = np.random.default_rng(0)
    img = np.zeros((224, 224, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:224, 0:224]
    img[(xx - 112) ** 2 + (yy - 112) ** 2 < 70 ** 2] = [0.9, 0.75, 0.1]     # 一个"黄色圆盘"
    from PIL import Image as PILImage

    img_t = preprocess(PILImage.fromarray((img * 255).astype(np.uint8))).unsqueeze(0)

    with torch.no_grad():
        prob = torch.softmax(net(img_t), dim=-1)[0]
    top5 = prob.topk(5)
    print("\n  top-5 预测（合成图，仅演示管线）：")
    for v, i in zip(top5.values, top5.indices):
        print(f"    {categories[int(i)]:<28}{float(v):.4f}")
    print("""
  换成真实图片的做法：
    from PIL import Image
    img = Image.open("cat.jpg").convert("RGB")
    img_t = preprocess(img).unsqueeze(0)
    """)

section("5) 迁移学习：真实项目里最常见的用法")
if net is not None:
    for p in net.parameters():                  # ① 冻结骨干
        p.requires_grad_(False)
    net.fc = nn.Linear(net.fc.in_features, 10)  # ② 换掉分类头
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"  冻结骨干 + 替换分类头后：可训练参数 = {human_num(float(trainable))} "
          f"（占总参数 {trainable / total:.2%}）")
    print("""
  三种迁移策略：
    ① 冻结骨干，只训练新 head（数据少时首选，几分钟就能训完）
    ② 全部微调（数据多、分布差异大）
    ③ 分层学习率：骨干用小 lr，head 用大 lr（最常用）
  LLM 场景的对应物就是 LoRA（第 5 章第 7 节）：冻结全部，只训低秩增量。
""")

section("6) 现代视觉模型怎么选")
print("""
  ResNet-18/50      : 基线，快，工业部署友好（TensorRT 优化成熟）
  ConvNeXt          : 用 ViT 的训练技巧重做的 CNN，精度/速度都很好
  ViT / Swin        : 数据量大时更强（第 6 章第 8 节）
  EfficientNet / MobileNet : 端侧、算力受限
  torchvision 一行切换：models.convnext_tiny(weights=...) / models.vit_b_16(weights=...)
""")
