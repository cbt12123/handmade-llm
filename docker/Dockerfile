# =============================================================================
# LLM_Line 完整环境（推荐）
#   - 基于 nvidia/cuda devel 镜像：自带 nvcc，可直接编译第 7 章的 CUDA 算子
#   - 需要宿主机安装 NVIDIA Driver + nvidia-container-toolkit
#
# 构建：
#   docker build -f docker/Dockerfile -t llm-line:cu121 .
# 运行：
#   docker run --rm -it --gpus all -v "$PWD":/workspace -w /workspace llm-line:cu121 bash
# =============================================================================
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0" \
    # HF 模型/数据集的缓存目录（挂 volume 可复用；国内可改成 HF_ENDPOINT=https://hf-mirror.com
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1

# 1) 系统依赖：编译 CUDA / C++ 扩展用
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        build-essential cmake ninja-build \
        git curl vim ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 2) Python 依赖（torch 指定 CUDA 12.1  wheel）
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir \
        torch==2.1.2 torchvision==0.16.2 \
        --index-url https://download.pytorch.org/whl/cu121 \
    && python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

# 3) 代码
WORKDIR /workspace
COPY . /workspace

# 4) 冒烟自检（构建时可选；不想跑可注释掉）
RUN python3 -c "import torch, numpy; print('torch', torch.__version__, 'cuda', torch.version.cuda)" \
    && nvcc --version | tail -n 2 \
    && python3 -c "import transformers, datasets, peft; print('transformers', transformers.__version__)"

# 5) 可选项：预装 flash-attn（编译较慢，约 10 分钟；不需要可注释掉）
# RUN python3 -m pip install --no-cache-dir flash-attn --no-build-isolation

CMD ["bash"]
