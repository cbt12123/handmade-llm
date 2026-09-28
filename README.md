# LLM_Line：从 Python 循环到 NVIDIA 算子的完整路线

> 面向**只会 Python 基础语法（循环 / 函数 / 类）**的读者，一路讲到能自己写 CUDA 算子。
> 全程 `md + 可运行代码`：每一章既有讲义（**含大量可直接抄的代码块与真实运行输出**），
> 也有配套的 `script/NN_xxx/` 脚本目录。讲义里出现的每一段输出，都来自脚本的真实运行结果。

---

## 0. 三句话说明这份材料怎么用

1. **按章节顺序读**：`docs/01` → `docs/07`，每章读完立刻跑 `script/` 下对应目录的脚本。
2. **每个脚本都能独立运行**：`python3 script/04_transformer/01_attention_numpy.py`，不需要联网、不需要下载数据集（全部用合成数据）。
3. **没有 GPU 也能学完 1~6 章**；第 7 章的 CUDA 脚本会自动降级为 CPU 模拟（打印 `[SKIP]`）。

```bash
# 一键冒烟测试（跑完所有章节的所有脚本）
bash script/run_all.sh            # 全部
bash script/run_all.sh 04         # 只跑第 4 章
```

---

## 1. 学习路线图

```
        ┌───────────────────────────────────────────────────────────────┐
        │ 第 1 章  机器学习基础                                          │
        │   梯度下降 → 线性/逻辑回归 → 手写 MLP 反向传播 → 正则化        │
        │   关键产出：你会手写反向传播，并理解 PyTorch 在替你做什么       │
        └───────────────────────────────────────────────────────────────┘
                                    ↓
        ┌───────────────────────────────────────────────────────────────┐
        │ 第 2 章  CNN（局部性 + 权值共享）                              │
        │   手写 conv2d / im2col / 池化 / 卷积反向传播 → 看卷积核长啥样  │
        └───────────────────────────────────────────────────────────────┘
                                    ↓
        ┌───────────────────────────────────────────────────────────────┐
        │ 第 3 章  NLP 基础                                              │
        │   分词(BPE) → n-gram → Word2Vec → HMM/Viterbi → RNN/LSTM      │
        │   → Seq2Seq + Attention（通往 Transformer 的最后一级台阶）     │
        └───────────────────────────────────────────────────────────────┘
                                    ↓
        ┌───────────────────────────────────────────────────────────────┐
        │ 第 4 章  Transformer                                          │
        │   注意力 → 多头 → 位置编码 → Block → 掩码 → 训练 → 可视化      │
        └───────────────────────────────────────────────────────────────┘
                                    ↓
        ┌───────────────────────────────────────────────────────────────┐
        │ 第 5 章  LLM 核心技术（推理与训练工程）                        │
        │   显存/Token 预算 → 采样 → KV Cache → RoPE → 量化             │
        │   → FlashAttention → LoRA → 连续批处理/PagedAttention          │
        └───────────────────────────────────────────────────────────────┘
                                    ↓
        ┌───────────────────────────────────────────────────────────────┐
        │ 第 6 章  大模型架构全景（进阶）                                │
        │   BERT / T5 / GPT / Llama / MoE / Mamba / GQA·MLA / 多模态     │
        │   + 训练并行栈（DP·TP·PP·ZeRO）                                │
        └───────────────────────────────────────────────────────────────┘
                                    ↓
        ┌───────────────────────────────────────────────────────────────┐
        │ 第 7 章  NVIDIA 算子编写（提升）                               │
        │   CUDA 基础 → PyTorch extension → 融合算子 → GEMM 分块        │
        │   → Triton → 性能分析(Roofline/nsys/ncu)                       │
        └───────────────────────────────────────────────────────────────┘
```

---

## 2. 目录结构

```
LLM_Line/
├── README.md                 ← 你在这里（总纲）
├── docs/                     ← 各章讲义（理论 + 公式 + 代码索引）
│   ├── 01-机器学习基础.md
│   ├── 02-卷积神经网络.md
│   ├── 03-NLP基础.md
│   ├── 04-Transformer.md
│   ├── 05-LLM核心技术.md
│   ├── 06-大模型架构全景.md
│   ├── 07-NVIDIA算子编写.md
│   └── 附录-术语表与公式速查.md
├── script/                   ← 可运行代码（按章节分子目录）
│   ├── common_utils.py       ← 共用小工具（种子、软依赖探测、ASCII 热力图、grad check）
│   ├── run_all.sh            ← 一键冒烟测试
│   ├── 01_ml_basics/         （8 个脚本）
│   ├── 02_cnn/               （6 个）
│   ├── 03_nlp/               （6 个）
│   ├── 04_transformer/       （7 个）
│   ├── 05_llm/               （8 个）
│   ├── 06_architectures/     （10 个）
│   └── 07_nvidia_kernels/    （CUDA / Triton / 性能分析 + CPU 模拟）
├── docker/
│   ├── Dockerfile            ← GPU 版（nvidia/cuda devel，自带 nvcc）
│   ├── Dockerfile.cpu        ← 无显卡时的轻量版
│   └── docker-compose.yml
└── requirements.txt
```

---

## 3. 环境搭建（两种方案）

依赖分三层，**缺哪层都不会让脚本报错，只会优雅降级（打印 `[SKIP]`）**：

| 层 | 依赖 | 覆盖内容 |
|---|---|---|
| 手写层（必需） | `numpy`、`torch` | 第 1~6 章全部手写实现 |
| 真实生态层（推荐） | `transformers`、`tokenizers`、`datasets`、`peft`、`accelerate`、`scikit-learn`、`torchvision` | 各章的 `real_*` 脚本：真实模型、真实 tokenizer、真实数据集、真实 LoRA |
| GPU 算子层（可选） | `nvcc`（CUDA Toolkit devel）、`triton`、`flash-attn` | 第 7 章编译与运行 CUDA/Triton kernel |

```bash
pip install -r requirements.txt
# 真实生态部分需要联网下载模型（首次约 1GB，之后走缓存）
export HF_ENDPOINT=https://hf-mirror.com      # 国内网络建议加这一行
```

### 方案 A：Docker（推荐，一次到位）

```bash
# 有 NVIDIA 显卡（需要已安装驱动 + nvidia-container-toolkit）
docker build -f docker/Dockerfile -t llm-line:cu121 .
docker run --rm -it --gpus all -v "$PWD":/workspace -w /workspace llm-line:cu121 bash

# 没有显卡（跑 1~6 章 + 第 7 章的 CPU 模拟）
docker build -f docker/Dockerfile.cpu -t llm-line:cpu .
docker run --rm -it -v "$PWD":/workspace -w /workspace llm-line:cpu bash
```

进入容器后：

```bash
python3 script/07_nvidia_kernels/00_cuda_env_check.py   # 确认 GPU / 算力 / nvcc
bash script/run_all.sh                                   # 跑通全部章节
```

### 方案 B：本机 conda / venv

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# torch 请按你的 CUDA 版本从 https://pytorch.org 选用对应命令，例如：
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

最低依赖其实只有 **numpy**（第 1~3 章大部分脚本纯 numpy 就能跑）；
**torch** 从第 4 章开始必需；**nvcc** 只在第 7 章需要。

---

## 4. 各章速览（先看这段，再决定从哪里开始）

| 章节 | 讲义 | 代码目录 | 学完你能做什么 |
|---|---|---|---|
| 1 机器学习基础 | `docs/01-机器学习基础.md` | `script/01_ml_basics/` | 手写梯度下降与反向传播，看懂 PyTorch 训练循环 |
| 2 CNN | `docs/02-卷积神经网络.md` | `script/02_cnn/` | 手写 conv2d、im2col、卷积反向传播，理解"局部性 + 权值共享" |
| 3 NLP 基础 | `docs/03-NLP基础.md` | `script/03_nlp/` | 手写 BPE、Word2Vec、HMM+Viterbi、RNN/LSTM、Seq2Seq+Attention |
| 4 Transformer | `docs/04-Transformer.md` | `script/04_transformer/` | 从零搭 Transformer 并训练它，看懂注意力图 |
| 5 LLM 核心技术 | `docs/05-LLM核心技术.md` | `script/05_llm/` | 算显存/Token 预算，实现采样、KV Cache、RoPE、量化、LoRA |
| 6 大模型架构 | `docs/06-大模型架构全景.md` | `script/06_architectures/` | 讲清 BERT/T5/GPT/Llama/MoE/Mamba/MLA/多模态各自是什么 |
| 7 NVIDIA 算子 | `docs/07-NVIDIA算子编写.md` | `script/07_nvidia_kernels/` | 写 CUDA kernel 并接进 PyTorch，做性能分析与优化 |

> 每章的 `script/` 目录末尾都有 `real_*` 脚本：**用真实库（sklearn / torchvision /
> transformers / datasets / peft）重做一遍本章的手写实现**。这部分需要联网下载模型或数据集
> （首次约 1GB，之后走缓存），缺库或离线时会打印 `[SKIP]`，不影响其余脚本。

---

## 5. 贯穿全书的三条主线

1. **一切都是同一个循环**：`前向 → 损失 → 反向 → 更新`。
   第 1 章手写一次，后面只是把"模块"换成了 Conv / Attention / MoE / 自定义 CUDA 算子。
2. **深度学习工程 ≈ 与显存和带宽做斗争**。
   KV Cache、量化、融合算子、PagedAttention、GEMM 分块，看似不相关，本质都是"少读一点显存"。
3. **先让它跑起来，再让它正确，最后才让它快**。
   每个脚本都遵循：`参考实现 → 数值对比 → 性能基准`（第 7 章把它写成了固定流程）。

---

## 6. 常见疑问

**Q：数据从哪来？需要联网下载吗？**
A：不需要。全部使用合成数据（随机生成的图形、规则生成的语料、模板造的句子），
好处是脚本 100% 可复现、几秒钟就能跑完，缺点是"效果数字"不代表真实任务水平 ——
真实训练请换成你自己的数据集。

**Q：为什么有的章节用 numpy，有的用 torch？**
A：numpy 版是为了"把公式变成你看得见的循环"（第 1~4 章大量保留）；
torch 版是为了"跑得动、能接工程"。两边都做了**数值对照**（`check_close`），
你可以确认自己手写的和框架算出来的完全一致。

**Q：没有显卡第 7 章怎么学？**
A：`script/07_nvidia_kernels/08_cpu_simulation.py` 用纯 Python 模拟了内存合并、
bank 冲突、warp 分化、occupancy 四个核心机制；其余脚本会打印 `[SKIP]` 并把
CUDA 源码里的知识点用文字与 CPU 代码讲清楚。等你有机器了再回来编译。

**Q：脚本跑一半报错怎么办？**
A：先看报错里是否含 `[SKIP]`（那是正常的降级）；否则通常是依赖缺失或 dtype 不匹配
（numpy 默认 float64，torch 默认 float32）。`bash script/run_all.sh <章号>` 可以定位到具体脚本。

---

## 7. 建议的学习节奏

- **第 1 周**：第 1~2 章。目标：能默写出带反向传播的两层 MLP 与 conv2d。
- **第 2 周**：第 3~4 章。目标：能默写出多头注意力，并解释 Q/K/V 分别是什么。
- **第 3 周**：第 5 章。目标：能估算"7B 模型 + 32k 上下文 + batch 8 要多少显存"。
- **第 4 周**：第 6 章。目标：能在白板上画出 Llama 一层，并说出每个组件为什么这样设计。
- **第 5 周及以后**：第 7 章 + 自己的项目。目标：能写一个融合算子并给出加速比。

> 每章结尾都有"结论速记 / 选型建议"，赶时间时可以只读那些段落，再回头补代码。
