# 第10章 PyTorch训练与模型管理

目标：把上章手写梯度换成自动求导，建立可训练、可评价、可保存、可恢复的完整流程。自动求导负责计算梯度，数据划分与评价仍由你负责。

## 10.1 使用conda与VS Code建立独立环境

本部分固定一组教学版本，便于复现，不要求追随最新版。以下命令在PowerShell或Anaconda Prompt执行；若PowerShell尚未初始化conda，可先在Anaconda Prompt运行。

```powershell
conda create -n handmade-ml python=3.12 pip -y --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda activate handmade-ml
python -m pip install numpy==1.26.4 matplotlib==3.8.4 scikit-learn==1.5.2 pillow joblib -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

CPU版本打印`False`是预期结果。本部分基础实验全部可在CPU完成。通用Python包使用清华PyPI镜像；conda创建使用清华Anaconda镜像；PyTorch使用官方CPU索引，二者不要混为一谈。镜像不可用时，普通库去掉`-i`改用默认PyPI，conda渠道换为官方`defaults`；版本仍保持一致。[清华PyPI说明](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/) · [官方版本组合](https://pytorch.org/get-started/previous-versions/)

VS Code按`Ctrl+Shift+P`，选择`Python: Select Interpreter`，选`handmade-ml`。新建终端后检查`python -c "import sys; print(sys.executable)"`，确认确实是该环境，而不是base。GPU安装涉及驱动与CUDA组合，后面部署和算子部分再展开；需要自行启用GPU时按[官方安装页](https://pytorch.org/get-started/locally/)选择对应组合。

## 10.2 Tensor的形状、类型与设备

Tensor是可参与自动求导的多维数组。分类输入通常是float32，类别编号通常是int64（`torch.long`），模型与输入应在同一设备。

```python
import torch
x = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float32)
y = torch.tensor([0, 1], dtype=torch.long)
print(x.shape, x.dtype, x.device)
print(x @ torch.ones(2, 1))  # (2, 1)
```

`reshape`改变观察形状，不能替代维度交换；交换图像通道轴应使用`permute`。`.numpy()`要求CPU张量且不参与梯度，常用`tensor.detach().cpu().numpy()`。

## 10.3 自动求导与计算图

```python
x = torch.tensor([2., 3.], requires_grad=True)
loss = x.square().sum()
loss.backward()
print(x.grad)  # tensor([4., 6.])
```

公式是$L=x_1^2+x_2^2$，梯度$(2x_1,2x_2)$。`requires_grad=True`让框架跟踪相关运算，`backward()`沿计算图运用链式法则。它不是数值试探，也不会替你决定损失是否合理。

非标量输出的反向传播需要指定上游梯度；初学先把批量损失求平均成标量。动态图每次前向重新建立，默认反向后释放中间缓存。

## 10.4 梯度累积、清零与停止求导

`.grad`默认累加，多次调用反向不会覆盖。一般每个更新步骤之前用`optimizer.zero_grad()`清除旧梯度。刻意做梯度累积时，则在多个小批量之后更新，且要正确缩放损失。

`detach()`把某个值从图上断开；`torch.no_grad()`让一段计算不建立梯度图。不要用`loss.item()`计算后续损失：它得到Python数值，梯度链已经断开。

## 10.5 Module与参数注册

```python
from torch import nn
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(64, 10))
    def forward(self, x):
        return self.net(x)
```

`nn.Linear(64,10)`把每条64维输入变成10个logit。赋给`self`的子模块会注册参数，`model.parameters()`才能交给优化器。动态创建多层应使用`nn.ModuleList`或`nn.Sequential`，普通Python列表不会自动注册其中的子模块。

## 10.6 Dataset与DataLoader

```python
from torch.utils.data import TensorDataset, DataLoader
dataset = TensorDataset(torch.randn(100, 64), torch.randint(0, 10, (100,)))
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
xb, yb = next(iter(loader))
print(xb.shape, yb.shape)  # (16, 64), (16,)
```

Dataset定义“一条数据怎么取”，DataLoader负责组成批次。训练常打乱；评价无需打乱。本教程Windows示例使用`num_workers=0`，避免初学阶段多进程入口与环境问题。

## 10.7 一个完整训练步骤

![训练步骤](../images/part03-training.png)

```python
model = Classifier()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
model.train()
for xb, yb in loader:
    logits = model(xb)
    loss = nn.functional.cross_entropy(logits, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

这里logit是$(B,10)$，标签$(B,)$，损失是标量。清零放在前向前或反向前都可以，关键是反向前不存在不希望累积的旧梯度。`backward()`计算梯度，`step()`才修改参数。

## 10.8 train、eval与no_grad各做什么

`model.train()`启用训练行为，例如Dropout；`model.eval()`切换评价行为，例如关闭Dropout、让BatchNorm使用保存的统计量。`eval()`本身不会关闭求导，评价通常还需`no_grad()`。

```python
model.eval()
with torch.no_grad():
    logits = model(torch.zeros(3, 64))
    prediction = logits.argmax(dim=1)
```

不切换`eval`，同一个输入可能因Dropout得到不同输出。反过来，训练时遗忘`train`也会改变你实际训练的模型行为。

`argmax(dim=1)`选每条样本中分数最高的类别编号。若只需要类别，不必先Softmax，因为Softmax不会改变同一行各类别的大小次序。

## 10.9 验证曲线、最佳模型与早停

每轮训练后在固定验证集评价，记录训练与验证损失。参考脚本的“训练损失”也在`eval`模式下对整份训练集重算，因此比训练时混合Dropout的小批量损失更便于与验证曲线比较。

验证损失最低时复制参数：`best = copy.deepcopy(model.state_dict())`。仅保存引用可能随下一轮训练改变。早停是在连续若干轮未改善后停止训练；本章参考实现固定25轮并恢复验证最佳参数，**没有实现提前停止循环**。[保存模型说明](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)

## 10.10 常见错误与排查顺序

| 现象 | 先检查 |
|---|---|
| 矩阵不能相乘 | 输出末维是否等于下一层输入维度 |
| 标签类型错误 | 多分类标签是否long，范围是否0～C-1 |
| loss为NaN | 输入是否有限、学习率是否过大、是否手工log(0) |
| loss不下降 | 参数是否注册、是否step、是否断开计算图 |
| 验证结果忽高忽低 | 是否eval、验证集是否固定、样本是否太少 |

先尝试在极小训练集上拟合，确认流程能学习；再扩大数据。不要用测试集排查并选择大量配置。

## 10.11 保存权重、恢复训练与推理

推理只需要模型结构、最佳权重和预处理约定。恢复训练还需要最后一轮权重、优化器状态、轮数和随机状态。最佳权重与最后权重可能不同，不能把最佳模型和另一轮优化器状态随意拼接。

```python
torch.save(model.state_dict(), "best.pt")
loaded = Classifier()
loaded.load_state_dict(torch.load("best.pt", weights_only=True))
loaded.eval()
```

本章保存`10_best.pt`用于推理，`10_resume.pt`用于恢复，包含AdamW状态与CPU随机状态。脚本验证恢复后再更新一个批次能得到有限损失；它不宣称逐位复现完整训练轨迹。复杂精确续训还需DataLoader、采样顺序、其他随机数生成器以及GPU随机状态。

## 10.12 章末产出：训练、加载与续训实验

```powershell
python script/part03/10_torch_training.py
```

产出是训练曲线、最佳权重、最后训练检查点和报告。输入使用第8章相同digits划分，像素固定除以16，不从测试集拟合统计量。

![MLP训练与验证曲线](../images/part03-result-10_curves.png)

报告还记录第9章同结构NumPy梯度与PyTorch自动求导的最大绝对差，要求小于`1e-10`；比较使用float64小批量，正式分类训练使用float32。

验收：自动求导得到$(4,6)$；重新加载最佳权重预测完全一致；恢复优化器后可更新一个批次；能解释`eval`与`no_grad`的不同职责。

**检查：**有`loss.backward()`就训练完成了吗？答案：还没有；它只把梯度写入参数，更新需要优化器`step()`。

[上一章](09-模型如何学习与神经网络.md) · [返回目录](README.md) · [下一章](11-CNN与图像识别.md)
