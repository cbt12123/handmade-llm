# 第四部分：大模型与部署

本部分从第13章的序列生成进入真实开放权重模型。先理解语言模型与结构配置，再加载现成权重，分析生成和KV Cache，估算资源，最后完成HTTP服务与评价。

正文直接讲解概念与计算，局部公式先给定义和例子，组合过程按依赖顺序展开。每章都有可运行脚本和可复核产出，不要求重新预训练大模型。

## 章节与产出

| 章节 | 内容 | 产出 |
| --- | --- | --- |
| [14 语言模型与大模型结构](14-语言模型与大模型结构.md) | 语言模型目标、训练阶段、GQA与模型配置 | 真实配置、张量形状和环境报告 |
| [15 本地加载与对话输入](15-本地加载与对话输入.md) | 文件、分词、消息模板、权重加载与生成 | 本地回答、token与候选概率报告 |
| [16 生成策略与KV缓存](16-生成策略与KV缓存.md) | 贪心、温度、top-p、prefill与decode | 缓存等价性和逐步耗时 |
| [17 资源预算、量化与参数适配](17-资源预算量化与参数适配.md) | 权重与KV内存、INT8小矩阵、LoRA合并 | 预算表、量化误差与合并验证 |
| [18 HTTP服务与部署实践](18-HTTP服务与部署实践.md) | 常驻模型、接口校验、并发限制、容器引擎 | HTTP服务、客户端和状态码验证 |
| [19 提示设计、评价与部署交付](19-提示设计评价与部署交付.md) | 输出契约、任务评价、延迟与文件版本 | 原始失败记录、评价报告与manifest |

图片在根目录`images/`，生成源码为[tools/draw_part04.py](../tools/draw_part04.py)。实验在`script/part04/`，报告在`outputs/part04/`，教学评价数据在`data/part04/`。

## 环境与模型准备

先做[第四部分入口自测](../附录/02-学习路线与前置自测.md#5-第四部分入口从训练转到加载与生成)：区分位置与词表轴、条件概率乘法、训练与推理。这里加载现有权重，不延续第三部分的训练任务；生成会改变 token 与缓存，不默认更新权重。

首次安装 WSL、Docker，或尚未准备模型时，先看[统一环境准备指南](../附录/01-环境准备与复现.md)。下面区分作者已有环境与读者新建环境，不要求你的电脑预先存在名为 `vllm` 的 conda 环境。

继续使用VS Code编辑Markdown与源码。已有环境和模型优先复用，不根据环境名字判断功能是否可用。

本机检查到：Windows conda的`vllm`环境提供Python 3.10.20、PyTorch 2.10.0+cpu、Transformers 4.57.6、FastAPI 0.135.1与Uvicorn 0.41.0。其CPU路径用于结构、分词、缓存和自定义HTTP教学实验；WSL2 Docker中的现成vLLM镜像用于GPU引擎部署。两条路线的库版本、精度和协议分别记录，不能混称。

### 复用本地conda环境

```powershell
conda activate vllm
python -c "import sys,torch,transformers; print(sys.executable); print(torch.__version__,torch.version.cuda,torch.cuda.is_available()); print(transformers.__version__)"
$env:LLM_MODEL_PATH = 'G:\models\qwen2_5_1.5b_instruct'
$env:LLM_CPU_THREADS = '4'
```

这是本机已有模型目录，其他电脑修改为自己的路径。所有本地推理脚本只读取文件，默认不联网下载模型。运行脚本前应在同一个终端设置变量；新终端不会自动继承另一个终端临时设置的值。

初次真实加载发现torch 2.10.0与torchvision 0.20.1、torchaudio 2.5.1不匹配，触发`torchvision::nms`错误。本机已只修复配套附属库，保留原CPU版torch：

```powershell
python -m pip install torchvision==0.25.0 torchaudio==2.10.0 --no-deps --index-url https://download.pytorch.org/whl/cpu
```

这条命令仅适用于这里的torch 2.10.0 CPU组合，不要求其他读者不经检查直接执行。官方轮子与版本配套见[PyTorch版本表](https://pytorch.org/get-started/previous-versions/)。本机Windows环境虽然名为`vllm`，不作为原生Windows vLLM GPU运行环境。

### 没有可复用环境时

以下是重建课程CPU路径的固定版本示例，不需要已具备环境的读者重复安装：

```powershell
conda create -n handmade-llm python=3.10 -y
conda activate handmade-llm
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install transformers==4.57.6 fastapi==0.135.1 uvicorn==0.41.0 requests==2.32.5 numpy==2.2.6 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

后续章节的启动命令统一以 `handmade-llm` 为例。若复用本机已有的 `vllm` 环境，将对应激活命令换成 `conda activate vllm`；环境名称不会改变脚本或服务协议。

清华PyPI镜像用于普通Python包；PyTorch CPU/CUDA轮子使用官方对应索引。已有模型时不需要Hub下载镜像。首次下载失败时先检查代理或模型平台连接，不能把Python包镜像当成模型权重镜像。

当前基础模型为[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)。已有完整权重时直接使用；没有时可明确执行：

```powershell
python script/part04/download_model.py
```

首次会下载约3GB权重及分词器文件，放在`models/qwen2.5-1.5b-instruct`。该目录被Git忽略。使用此默认目录可不设置`LLM_MODEL_PATH`；正文第15章的短示例直接读取该变量，可设置为自己的完整路径。

## 基础实验顺序

从工作区根目录运行：

```powershell
python script/part04/14_model_structure.py
python script/part04/15_local_inference.py
python script/part04/16_generation_cache.py
python script/part04/17_memory_quantization.py
```

14只读配置，15和16分别加载真实模型，17操作小矩阵并读取配置。CPU路径使用FP32，权重内存约6.17GB，加载和运算还要留额外空间。生成速度取决于CPU与内存，短输出实验优先。

### 自定义HTTP教学服务

终端一保留模型路径变量并启动：

```powershell
python script/part04/18_server.py
```

终端二使用同样环境调用：

```powershell
python script/part04/18_client.py
python script/part04/19_evaluate.py
```

CPU服务监听`127.0.0.1:8000`，一个常驻模型、一次一个请求、总token预算512、输出上限128。非流式返回。启动后可访问`/health`；结束时在服务终端按`Ctrl+C`。

### 文件版本与交付

```powershell
python script/part04/19_release_manifest.py
```

这条命令需要读取模型文件计算哈希，但不复制或上传权重。许可与来源随模型记录，完整实验成绩保存为JSON。

## WSL2与现成GPU镜像

本机Ubuntu WSL2中已有Docker和`vllm/vllm-openai:v0.20.1-cu129-ubuntu2404`镜像，也能识别RTX 4060 Ti 16GB。先进入Ubuntu终端检查：

```bash
docker image ls
docker container ls -a
nvidia-smi
docker run --rm --gpus all --entrypoint python3 \
  vllm/vllm-openai:v0.20.1-cu129-ubuntu2404 \
  -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

容器的`--gpus all`与CUDA检查需要成功，只有WSL中的`nvidia-smi`成功还不足以证明容器能用GPU。镜像原始入口是`vllm serve`，检查Python时需要覆盖入口；Ubuntu镜像里的命令可能叫`python3`，不能假设一定有`python`别名。

以下从WSL终端启动一个独立教学容器，使用已有权重，不下载模型：

```bash
docker run --name handmade-llm-part04-vllm \
  --gpus all --ipc=host \
  -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 \
  -p 127.0.0.1:8001:8000 \
  -v /mnt/g/models/qwen2_5_1.5b_instruct:/models/qwen:ro \
  vllm/vllm-openai:v0.20.1-cu129-ubuntu2404 \
  /models/qwen --served-model-name tutorial-qwen \
  --host 0.0.0.0 --port 8000 --dtype float16 \
  --max-model-len 512 --max-num-seqs 4 \
  --gpu-memory-utilization 0.4 --enforce-eager
```

路径是本机映射示例，应替换成自己的模型路径。模型目录只读挂载。容器内部监听所有接口，但宿主端口只发布到回环地址，Windows调用端使用8001，避免与CPU教学服务的8000混淆。这里前台启动，保留Ubuntu终端查看日志，再从另一个Windows终端调用。

`--enforce-eager`先省去图捕获与编译优化，便于完成基础部署；不是最优性能配置。显存比例、最大长度和并发数是资源设置，不能解释成模型质量参数。

```bash
docker logs --tail 80 handmade-llm-part04-vllm
docker stop handmade-llm-part04-vllm
docker start handmade-llm-part04-vllm
```

同名容器已存在时用`docker start`恢复，不重复创建；需要保持前台并查看实时输出时用`docker start -a handmade-llm-part04-vllm`。本机首次后台启动曾无日志退出，保持WSL终端并以前台方式重启后完成了验证。WSL自身运行状态也应检查，不能只根据Docker返回容器ID判断服务就绪。

同一端口已有服务时先检查归属，不要为了跑课程停止其他项目的容器。

在Windows conda终端调用GPU引擎。HTTP客户端只需要requests，评价另用NumPy；本机`base`已有这些库，无需在Windows再装GPU推理环境：

```powershell
conda activate base
python script/part04/18_vllm_client.py
python script/part04/19_evaluate.py --backend vllm --url http://127.0.0.1:8001
python tools/verify_part04_vllm.py
python tools/record_part04_container.py
```

客户端使用`/v1/chat/completions`协议，检查完整回答、SSE流式事件和JSON schema约束。它与自定义服务的`/generate`客户端分开。GPU评价保存为`19_evaluation_vllm.json`，不覆盖CPU基线。核验工具再次执行客户端与评价，再检查四个并发请求；不需要先重复运行全部命令。环境记录工具读取指定容器的镜像ID、命令和实际依赖。

本机镜像内实际为Python 3.12.13、PyTorch 2.11.0+cu129、Transformers 5.7.0、vLLM 0.20.1，CUDA可用。服务使用FP16、FlashAttention 2，模型加载约2.89GiB，KV Cache预分配约3.33GiB。日志显示的缓存理论容量不是实际请求并发上限，本次调度配置仍是`max-num-seqs=4`。

## 核验与页面预览

```powershell
python tools/verify_part04.py
```

使用第四部分环境运行。工具执行真实模型脚本、临时启动自定义HTTP服务、发送边界请求、运行评价和manifest，再检查正文Python例子。它只停止自己启动的服务；端口8000被占用会报错，请先结束自己的教学服务。

只复查已有报告和正文时：

```powershell
python tools/verify_part04.py --check-only
```

生成图片与HTML预览使用前面已有Matplotlib及预览依赖的`handmade-ml`环境，避免为推理环境再装绘图库：

```powershell
conda activate handmade-ml
python tools/draw_part04.py
python tools/review_markdown.py
```

## 已验证范围与限制

本机已验证真实本地CPU模型、分词往返、八步缓存等价性、小矩阵量化与LoRA合并、自定义HTTP请求和严格格式评价。量化与LoRA不是整模型训练或压缩实验。基础服务也没有流式生成、队列与生产级高并发调度。

CPU评价8条教学用例严格通过3条：JSON提取值正确但带围栏，缺失信息回答多句号。完整失败记录保留，不能只根据总分判断知识能力。GPU容器已验证完整响应、流式拼接一致与JSON schema约束输出，结果见`18_vllm_http.json`；不同后端和精度的评价分别保存。

本机核验完成后已停止教学容器，保留其配置；镜像及本地权重继续复用，使用前述`docker start -a`恢复。四个并发请求的功能结果、流式与schema检查见`verification_vllm.json`。本部分未做真实整模型量化、微调或大规模吞吐压测。

第四部分先完成模型加载、部署与评价。第五部分再进入Agent与知识组织，第六部分进入NVIDIA算子开发。
