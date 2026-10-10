# 第六部分：NVIDIA 算子开发

这一部分先用原生 CUDA 学执行与数据复用，再用 Triton 实现融合 RMSNorm，最终接入已有 Qwen 模型与 HTTP 服务。目标是掌握如何判断一个算子是否值得用于真实推理。

## 章节与产出

| 章 | 内容 | 产出 |
|---|---|---|
| [28 从推理瓶颈到GPU执行](28-从推理瓶颈到GPU执行.md) | 模型调用位置、线程与内存、局部收益上限 | 环境记录与成本估算 |
| [29 第一个CUDA程序与数据布局](29-第一个CUDA程序与数据布局.md) | C++最小语法、索引、尾部与异步执行 | 向量加法与内存检查 |
| [30 归约、共享内存与矩阵复用](30-归约共享内存与矩阵复用.md) | warp归约、同步、分块GEMM | 原生RMSNorm、GEMM与cuBLAS比较 |
| [31 用Triton实现融合RMSNorm](31-用Triton实现融合RMSNorm.md) | 融合、类型转换、布局与支持范围 | 实际模型用的算子与多基线比较 |
| [32 Softmax、贪心选择与少算一步](32-Softmax贪心选择与少算一步.md) | 稳定Softmax、浮点反例、最后位置logits | 有条件减少计算的独立实验 |
| [33 性能测量与瓶颈分析](33-性能测量与瓶颈分析.md) | 预热、同步、交替采样、Profiler | 实际GPU调用证据 |
| [34 把融合算子接入真实推理](34-把融合算子接入真实推理.md) | 可恢复替换、KV缓存、生成一致性 | 完整推理对照与原始数据 |
| [35 算子部署、评价与交付](35-算子部署评价与交付.md) | 默认融合HTTP服务、对照、边界与交付 | 可调用服务与复现清单 |

前置知识是第一部分的数组与函数、第二部分的矩阵与归约、第三部分的PyTorch，以及第四部分的模型加载、KV缓存与HTTP。不会 C++ 不妨先读第29章最小语法，再对照完整源码；不要求先学完另一门 C++ 课程。

## 实际采用的环境

编辑与CPU脚本继续使用 Windows、conda base、VS Code。GPU脚本在WSL2 Ubuntu的Docker容器中执行；不要在Windows的CPU版PyTorch上运行后，把结果写成GPU验证通过。

| 用途 | 本次环境 |
|---|---|
| 编辑、绘图与HTTP客户端 | Python 3.12.4、NumPy 1.26.4、Matplotlib 3.8.4、requests 2.32.2 |
| GPU | RTX 4060 Ti 16GB，计算能力8.9，驱动591.86 |
| 原生CUDA编译与内存检查 | 已有 cuda-ops-dev:latest，nvcc 12.4.131，compute-sanitizer |
| Triton、真实模型与服务 | 已有 vllm/vllm-openai:v0.20.1-cu129-ubuntu2404 |
| 容器Python库 | Python 3.12.13、PyTorch 2.11.0+cu129、Triton 3.6.0、Transformers 5.7.0 |
| 模型 | 本地 Qwen2.5-1.5B-Instruct，隐藏维度1536、28层、FP16 |

这里借用 vLLM 镜像中的库运行 Transformers，没有调用 vLLM 推理引擎。两个镜像分别执行独立程序，没有把CUDA 12.4编译的扩展链接进CUDA 12.9的Torch。编译器与运行库的兼容问题不能靠“都是CUDA”忽略。

`cuda-ops-dev:latest` 是作者已有的本地镜像名，不是公共仓库的必备镜像。首次准备时，按[环境准备与复现](../附录/01-环境准备与复现.md)配置 WSL、Docker 和 GPU，再用仓库的 [Dockerfile](../assets/part06/environment/Dockerfile)构建 `handmade-llm-cuda:12.4.1`。它提供 nvcc、C++ 编译器、cuBLAS 开发文件与 compute-sanitizer；runtime 镜像通常不包含完整编译工具。实际版本仍以自己的报告为准。

本次没有新增GPU环境、安装GPU库或下载模型。若base缺少绘图或客户端依赖，可使用清华镜像：

```powershell
conda activate base
python -m pip install numpy matplotlib requests -i https://pypi.tuna.tsinghua.edu.cn/simple
```

第三方镜像只是下载来源，不代表版本兼容性已经验证。GPU版Torch和Triton应按目标CUDA环境安装，不用这条CPU侧命令代替。

## 启动实验容器

下面使用作者本机路径，读者应替换成自己的目录。先在WSL执行 `docker images` 和 `docker ps -a` 检查已有镜像与容器。

```bash
docker run --name handmade-llm-part06-dev --gpus all --ipc=host \
  -v /mnt/g/project/handmade-llm-normal:/workspace \
  -w /workspace --entrypoint bash handmade-llm-cuda:12.4.1 \
  -lc 'sleep infinity'
```

这条命令在当前终端持续运行。另开终端执行检查：

已检查过作者原有 `cuda-ops-dev:latest` 的读者，也可将上面的镜像名替换成它；历史性能报告来自该原有镜像。

```bash
docker exec handmade-llm-part06-dev python3 script/part06/29_cuda_examples.py --sanitize
```

默认架构sm_89对应本次GPU，换卡后通过脚本的 `--arch` 参数指定合适架构。

接着启动运行容器，同样另开终端执行实验：

```bash
docker run --name handmade-llm-part06-runtime --gpus all --ipc=host \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v /mnt/g/project/handmade-llm-normal:/workspace \
  -v /mnt/g/models/qwen2_5_1.5b_instruct:/models/qwen:ro \
  -w /workspace --entrypoint bash \
  vllm/vllm-openai:v0.20.1-cu129-ubuntu2404 -lc 'sleep infinity'
```

`/models/qwen` 默认是模型目录；不同挂载目标可用容器环境变量 LLM_MODEL_PATH 指定。模型文件只读挂载，代码与小报告写入工作区。

```bash
docker exec handmade-llm-part06-runtime python3 tools/verify_part06_gpu.py
```

这个入口依次记录GPU环境、验证与测量算子、测试选择步骤、测量真实模型并剖析。模型缓存复用在同一进程内，避免两次重新加载。运行期间不要同时启动其他GPU推理服务，减少相互干扰。

## 各章脚本与报告

| 命令或文件 | 用途 |
|---|---|
| python script/part06/28_cost_model.py | CPU可执行的线程索引、逻辑字节与Amdahl估算 |
| 29_cuda_examples.py --sanitize | 编译3个CUDA程序、cuBLAS对照及内存检查 |
| 31_verify_and_benchmark.py | 多精度边界验证，eager/compile/Triton微基准 |
| 32_selection.py | argmax比较与浮点反例 |
| 34_model_benchmark.py | 3种长度的真实推理、最后logits独立对照 |
| 33_profile_model.py | 内核事件、回退次数与异常恢复 |
| 35_server.py / 35_client.py | 服务与实际HTTP验收，详见第35章 |
| 35_release.py | 源码和报告SHA-256交付清单 |
| tools/draw_part06.py | 从报告重建插图 |

脚本位于根目录 `script/part06/`，原生源码见[附件](../assets/part06/README.md)，小报告在 `outputs/part06/`，插图在 `images/part06-*.png`。

## 已验证的范围与结果

原生CUDA三个程序均通过内存检查。教学分块GEMM比朴素实现快，但仍慢于cuBLAS。融合RMSNorm通过三种精度、边界形状与尺度检查，真实模型剖析观察到228次自写内核调用、零回退。

这次真实生成的加速比为128 token输入1.162、512 token输入1.149、1024 token输入1.020；完整32步生成IDs一致。1024 token预填充加速比0.961，保留这项不利结果。HTTP三条提示词也分别比较了完整返回IDs与调用计数。

这些结论针对本次Transformers eager、单请求、FP16、贪心生成。没有证明优于成熟推理引擎，也没有实现训练、任意布局、多请求批处理或投机解码。

## 结束实验

在WSL停止本部分自己启动的容器，保留镜像与容器便于下次复用：

```bash
docker stop handmade-llm-part06-service handmade-llm-part06-runtime handmade-llm-part06-dev
```

只启动过其中一部分时，仅填写实际运行的名字。下一次可分别 `docker start` 实验容器，在另一个终端 docker exec；服务可 `docker start -a` 查看日志。
