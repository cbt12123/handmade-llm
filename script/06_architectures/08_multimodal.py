"""
08 · 多模态：ViT / CLIP / LLaVA 这条线
========================================
多模态 LLM 的标准三段式：
  ① 视觉编码器（ViT）  把图像切成 patch，编码成一串"视觉 token"
  ② 对齐模块（Projector）把视觉 token 映射到 LLM 的词嵌入空间
  ③ LLM 主干            把"视觉 token + 文本 token"当成一条序列做自回归

本脚本：
  1. 手写 ViT 的 patch embedding（一行 Conv2d 就够了）并跑通前向
  2. 手写 CLIP 的对比学习损失（InfoNCE），在合成图文对上训练，看检索准确率上升
  3. 手写 LLaVA 的 projector，把视觉特征对齐到"文本嵌入空间"并验证检索效果
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from common_utils import require_torch, section, set_seed, subsection, timer

set_seed(17)
torch = require_torch("08_multimodal")
nn = torch.nn
F = torch.nn.functional

# ----------------------------------------------------------------------------------
section("1) ViT：把图像变成 token 序列")


class PatchEmbedding(nn.Module):
    """图像 → patch 序列。用 Conv2d 实现最简洁：kernel=stride=patch_size。"""

    def __init__(self, img_size=32, patch_size=8, in_ch=1, d_model=64):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                     # (B, d, H/P, W/P)
        return x.flatten(2).transpose(1, 2)  # (B, num_patches, d)


class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=8, d_model=64, heads=4, layers=2, num_classes=4):
        super().__init__()
        self.patch = PatchEmbedding(img_size, patch_size, 1, d_model)
        n = self.patch.num_patches
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model,
                                               batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, img):
        B = img.shape[0]
        x = self.patch(img)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        x = self.encoder(x)
        return self.head(x[:, 0])            # 用 [CLS] token 做分类


vit = ViT()
img = torch.randn(2, 1, 32, 32)
out = vit(img)
print(f"  图像 {tuple(img.shape)} → {vit.patch.num_patches} 个 patch → logits {tuple(out.shape)}")
print("""
  patch_size=8, 32×32 图 → 16 个 patch（相当于 16 个 token）
  真实配置：ViT-L/14 用 224×224 图 + 14×14 patch → 256 个 token（+1 CLS）
  代价：patch 越多，LLM 的上下文被占得越多 → 所以有 "anyres" / token 压缩等技巧
""")

# ----------------------------------------------------------------------------------
section("2) CLIP：对比学习让图文对齐")
CLASSES = ["circle", "square", "triangle", "cross"]


def make_imtext_pairs(n=400, size=32, seed=0):
    """合成"图形-文本"对：图像是 4 种形状之一，文本是该形状的名字（用随机向量表示）。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    imgs, labels, texts = [], [], []
    for _ in range(n):
        c = int(rng.integers(0, 4))
        cx, cy = rng.integers(10, size - 10, 2)
        r = 7
        if c == 0:
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 < r**2
        elif c == 1:
            mask = (np.abs(xx - cx) < r) & (np.abs(yy - cy) < r)
        elif c == 2:
            mask = (yy > cy) & (np.abs(xx - cx) < (yy - cy) * 0.8) & (yy - cy < r * 1.5)
        else:
            mask = (np.abs(xx - cx) < 2) | (np.abs(yy - cy) < 2)
        img = mask.astype(np.float32) + rng.normal(0, 0.1, (size, size))
        imgs.append(img)
        labels.append(c)
        # 文本表示：类别名的 one-hot + 一点噪声（模拟句子 embedding）
        t = np.zeros(4, dtype=np.float32)
        t[c] = 1.0
        texts.append(t + rng.normal(0, 0.05, 4))
    return (torch.tensor(np.array(imgs), dtype=torch.float32)[:, None],
            torch.tensor(np.array(texts), dtype=torch.float32),
            torch.tensor(np.array(labels)))


IMG_TR, TXT_TR, LBL_TR = make_imtext_pairs(400, seed=0)
IMG_TE, TXT_TE, LBL_TE = make_imtext_pairs(100, seed=1)

img_enc = ViT(num_classes=32)                 # 视觉编码器 → 32 维特征
txt_enc = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, 32))


def clip_loss(img_feat, txt_feat, temperature=0.07):
    """InfoNCE：对称的图文对比损失。"""
    img_feat = F.normalize(img_feat, dim=-1)
    txt_feat = F.normalize(txt_feat, dim=-1)
    logits = img_feat @ txt_feat.T / temperature        # (N, N)
    labels = torch.arange(img_feat.shape[0], device=img_feat.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def retrieve_acc(img_enc, txt_enc, imgs, txts, labels):
    """图→文检索：注意测试集里同类别的文本有多条，所以要按【类别】判断是否正确。"""
    with torch.no_grad():
        fi = F.normalize(img_enc(imgs), dim=-1)
        ft = F.normalize(txt_enc(txts), dim=-1)
        sim = fi @ ft.T
        pred = sim.argmax(1)
    return (labels[pred] == labels).float().mean().item()


print(f"  训练前 图→文 top-1 检索准确率 = {retrieve_acc(img_enc, txt_enc, IMG_TE, TXT_TE, LBL_TE):.3f}")
opt = torch.optim.AdamW(list(img_enc.parameters()) + list(txt_enc.parameters()), lr=3e-3)
with timer("CLIP 训练耗时"):
    for step in range(201):
        loss = clip_loss(img_enc(IMG_TR), txt_enc(TXT_TR))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"  step {step:>4}  contrastive loss={loss.item():.4f}")
print(f"  训练后 图→文 top-1 检索准确率 = {retrieve_acc(img_enc, txt_enc, IMG_TE, TXT_TE, LBL_TE):.3f}")
print("""
  这就是 CLIP：不需要标注"这张图是什么"，只需要【配对的图文】，
  用对比学习把两个模态拉到同一个向量空间。
  为什么对比损失有效：正样本相似度要高于 batch 内所有负样本（batch 越大效果越好，
  所以 CLIP 用了 32768 的 batch；工程上常用梯度累积或类别内负样本共享来近似）。
""")

# ----------------------------------------------------------------------------------
section("3) LLaVA 的 Projector：把视觉特征『翻译』成 LLM 能懂的 token")
print("""
  LLaVA 的结构：
      ViT(视觉特征) → Projector(MLP 或 线性) → 拼到文本 token 前面 → LLM

  训练分两阶段：
    ① 预训练（只训 Projector）：大量图文对，冻结 ViT 和 LLM
    ② 指令微调（训 Projector + LLM）：视觉问答数据
  关键洞察：LLM 已经很懂语言了，只需要学会"看图说话"的映射，代价极低。
""")
# 玩具演示：假设 LLM 的词嵌入空间是 32 维，视觉特征是 32 维，训练一个 projector 对齐
torch.manual_seed(0)
llm_emb_space = torch.randn(4, 32)            # 4 个"词"的嵌入（对应 4 个类别名）
projector = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))

with torch.no_grad():
    vis_feat = img_enc(IMG_TR)[:200]          # 视觉特征（冻结）
target_emb = llm_emb_space[LBL_TR[:200]]      # 目标是学会映射到正确的词嵌入
opt2 = torch.optim.AdamW(projector.parameters(), lr=3e-3)
for step in range(300):
    pred_emb = projector(vis_feat)
    loss = F.mse_loss(pred_emb, target_emb)
    opt2.zero_grad()
    loss.backward()
    opt2.step()
with torch.no_grad():
    vis_te = img_enc(IMG_TE)
    sim = projector(vis_te) @ llm_emb_space.T
    acc = (sim.argmax(1) == LBL_TE).float().mean().item()
print(f"  Projector 训练后：视觉特征 → 词嵌入空间的 top-1 准确率 = {acc:.3f}")
print("  含义：projector 学会了把『一只猫的图』映射到『猫』这个词的嵌入附近，")
print("        之后 LLM 就能像处理普通文本 token 一样处理视觉 token。")

section("4) 多模态的工程要点")
print("""
  ① Token 预算：一张 1024×1024 图在高分辨率模式下可能有 1000+ 个视觉 token，
     会吃掉大半个上下文 → 需要 token 压缩（Q-Former / Perceiver Resampler / pixel shuffle）
  ② 分辨率：ViT 训练分辨率固定，推理时用更大分辨率要插值位置编码（或改用动态分辨率）
  ③ 训练稳定性：多模态训练比纯文本更容易 loss 尖峰 → 常用更小的 lr + 更多 warmup
  ④ 数据质量：图文对里的"描述与图不符"是最大的噪声源，清洗比堆量更重要
  ⑤ 视频：本质是"多帧图像"，时间维度要么拼进序列（token 爆炸），
     要么用 3D 位置编码 / 时间注意力压缩
""")
