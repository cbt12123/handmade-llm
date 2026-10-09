# 第六部分附件：原生CUDA源码与实现入口

正文逐步解释结构，附件提供能编译运行的完整源码。所有插图集中在根目录 images，结果集中在 outputs/part06。

| 文件 | 关注点 |
|---|---|
| [vector_add.cu](cuda/vector_add.cu) | 全局索引、尾部保护、错误检查与Event计时 |
| [rms_norm.cu](cuda/rms_norm.cu) | FP32教学归约、warp部分和、共享标量广播 |
| [matmul.cu](cuda/matmul.cu) | 朴素与分块GEMM、两次同步、cuBLAS对照 |
| [kernels.py](../../script/part06/kernels.py) | Triton向量加法、Softmax与真实融合RMSNorm |
| [model_adapter.py](../../script/part06/model_adapter.py) | 暂时替换模型方法、回退与finally恢复 |
| [inference.py](../../script/part06/inference.py) | 本地加载、聊天模板、缓存与贪心循环 |

原生RMSNorm用于理解CUDA归约，真实模型接入使用Triton版本，并按Qwen的精度转换约定计算。两者用途不同，不因为公式名字一样就忽略数据类型。

完整运行步骤见[第六部分指南](../../第六部分-NVIDIA算子开发/README.md)。源码是教学实现：未提供训练反向传播，未替代成熟矩阵乘法库，也未实现生产推理引擎的调度。
