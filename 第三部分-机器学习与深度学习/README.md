# 第三部分：机器学习与深度学习

前两部分提供了Python、数组、导数、矩阵、概率与序列的基础。本部分把这些知识连接成完整实验：任务与评价→学习机制→训练工程→CNN→NLP→Transformer。遇到公式先检查符号与形状，再算小例子，不必先背术语。

| 章节 | 学完能做什么 | 章末产出 |
|---|---|---|
| [8 机器学习任务与模型评价](08-机器学习任务与模型评价.md) | 正确划分、建立基线、评价分类 | digits分类器与测试报告 |
| [9 模型如何学习与神经网络](09-模型如何学习与神经网络.md) | 推导并检查两层网络梯度 | NumPy分类器、梯度检查、边界图 |
| [10 PyTorch训练与模型管理](10-PyTorch训练与模型管理.md) | 自动求导、训练、加载与恢复 | 最佳权重、续训检查点、曲线 |
| [11 CNN与图像识别](11-CNN与图像识别.md) | 计算卷积形状、训练与图片推理 | CNN、特征图、自备图预测 |
| [12 NLP与序列建模](12-NLP与序列建模.md) | 编码文本、比较基线、解释序列生成 | 中文教学分类与生成、真实短信扩展 |
| [13 注意力与Transformer](13-注意力与Transformer.md) | 理解QKV、掩码与自由生成 | 因果Transformer、注意力图、长度测试 |

## 怎么读

第一次按章节顺序，优先完成正文小例子与章末基础脚本。决策树、Word2Vec、HMM、im2col和迁移学习标为选读或扩展，不是继续阅读的门槛。第一次看到LSTM或Transformer组合公式时，沿每个局部计算走，不需要一次记住全式。

每章的目标与验收要求帮助你确定“目前是否学会”。如果向AI提问，建议贴具体输入形状、代码和报错，要求它解释当前步骤；拿到解释后自己运行一个小例子核对。

## 运行方式

环境统一conda+VS Code，安装与镜像说明见[10.1](10-PyTorch训练与模型管理.md#101-使用conda与vs-code建立独立环境)。第三部分独立环境叫`handmade-ml`，第一、第二部分的base不受影响。在VS Code打开工作区根目录并选择此解释器，然后运行：

```powershell
conda activate handmade-ml
python script/part03/08_sklearn_task.py
python script/part03/09_numpy_network.py
python script/part03/10_torch_training.py
python script/part03/11_cnn.py
python script/part03/12_text.py
python script/part03/13_transformer.py
```

基础实验无需下载数据或预训练权重，均可使用CPU。参考实现放在`script/part03/`，运行结果放在`outputs/part03/`，正文引用的图片统一在根目录`images/`。图片原始生成源码为`tools/draw_part03.py`；实验图由训练脚本生成，核验工具复制到`images/`供正文引用。

```powershell
python tools/draw_part03.py
python tools/verify_part03.py
```

完整实验已运行后，如果仅修改了正文或图片引用，可使用`python tools/verify_part03.py --check-only`执行正文短例子、检查本地链接并复查现有报告，不重新训练模型。

可选真实数据与迁移学习会首次联网下载：

```powershell
python script/part03/12_real_sms.py
python script/part03/11_transfer_learning.py
python tools/verify_part03.py --extras
```

`--extras`包含基础脚本并重新运行两个扩展，不必把以上三条都重复执行。短信缓存位于`data/part03/`，官方ResNet权重缓存由torch管理。原始短信来源与许可写在缓存目录README中。[UCI短信数据](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) · [官方ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)

## 数据与结论的边界

真实digits样本有1797张8×8图，不能当作MNIST或通用视觉能力。双月数据与中文小句子都是教学数据；中文小句子分数只检查实现。Transformer倒序任务训练长度固定，专门报告改变长度后的失败，不能据此称模型拥有通用推理能力。

本部分使用固定随机种子与CPU；不同库版本和设备可能造成数值差异。各章模型按验证集选择，再报告测试集结果。重复运行是复现已固定的实验；如果据测试成绩修改配置，原测试集就失去了最终独立评价用途。

全部章节为初稿，已通过程序与链接核验；初学者实际阅读体验仍需后续读者反馈验证。第四部分将在这些产出的基础上进入大模型与部署。

## 本机核验记录

本轮使用Windows、Python 3.12.15、NumPy 1.26.4、Matplotlib 3.8.4、scikit-learn 1.5.2、PyTorch 2.5.1+cpu与torchvision 0.20.1+cpu。完整依赖记录在`outputs/part03/requirements-verified.txt`；主命令的直接依赖版本已固定，其他依赖由安装工具解析。

本次MLP测试准确率约96.4%，CNN约96.1%；二者使用相同数据划分，但本轮没有做多随机种子显著性比较，因此不能凭这个差异宣称哪类架构更优。手写梯度与自动求导最大绝对差约`6.9e-18`。真实短信测试的spam召回率约93.0%，F1约92.2%，去重后的样本数为5159。

具体成绩、自由生成样例和通过的检查，以各章JSON与`outputs/part03/verification.json`为准；这是一轮固定实验的记录，不是对未来训练结果的保证。
