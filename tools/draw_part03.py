"""Original, reproducible learning diagrams; no downloaded illustrations."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'images'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':11})
def save(fig,name):
    fig.tight_layout(); fig.savefig(OUT/f'part03-{name}.png',dpi=160); plt.close(fig)
def flow(name,labels):
    fig,ax=plt.subplots(figsize=(11,2.4)); ax.set(xlim=(0,len(labels)),ylim=(0,1)); ax.axis('off')
    for i,label in enumerate(labels):
        ax.add_patch(Rectangle((i+.04,.3),.8,.4,facecolor='#e5efff',edgecolor='#3465a4')); ax.text(i+.44,.5,label,ha='center',va='center')
        if i<len(labels)-1: ax.annotate('',xy=(i+1.03,.5),xytext=(i+.85,.5),arrowprops={'arrowstyle':'->'})
    save(fig,name)
flow('workflow',['原始样本','划分数据','训练集拟合','验证集选方案','测试集评价'])
flow('backprop',['X → 仿射','A → tanh','H → 仿射','Z → sigmoid','损失 → 反向'])
flow('training',['取一批数据','前向与损失','清空梯度','反向求梯度','更新参数'])
flow('nlp',['文本','切分与编号','嵌入向量','序列模型','分类或生成'])
flow('transformer',['编号＋位置','多头注意力','残差与归一化','前馈网络','下个编号概率'])
fig,ax=plt.subplots(figsize=(5,4)); matrix=np.array([[45,5],[12,38]]); ax.imshow(matrix,cmap='Blues'); ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['预测负','预测正'],yticklabels=['真实负','真实正'],title='混淆矩阵：行是真实，列是预测')
for (r,c),v in np.ndenumerate(matrix): ax.text(c,r,str(v),ha='center',va='center')
save(fig,'confusion')
x=np.linspace(-6,6,200); fig,axes=plt.subplots(1,2,figsize=(9,3.4)); axes[0].plot(x,1/(1+np.exp(-x))); axes[0].set(title='Sigmoid',xlabel='z',ylabel='概率'); axes[1].plot(x,np.maximum(x,0),label='ReLU'); axes[1].plot(x,np.tanh(x),label='tanh'); axes[1].legend(); axes[1].set(title='激活函数',xlabel='输入'); save(fig,'activations')
fig,axes=plt.subplots(1,3,figsize=(9,3)); data=np.arange(25).reshape(5,5); kernel=np.array([[1,0,-1]]*3); result=np.array([[(data[i:i+3,j:j+3]*kernel).sum() for j in range(3)] for i in range(3)])
for ax,array,title in zip(axes,[data,kernel,result],['5×5 输入','3×3 核','3×3 输出']):
    ax.imshow(array,cmap='Blues'); ax.set_title(title); ax.axis('off')
    for (r,c),v in np.ndenumerate(array): ax.text(c,r,str(v),ha='center',va='center')
save(fig,'convolution')
fig,ax=plt.subplots(figsize=(5,4)); mask=np.tril(np.ones((7,7))); ax.imshow(mask,cmap='Blues',vmin=0,vmax=1); ax.set(title='因果注意力：蓝色可看，白色不可看',xlabel='被读取的位置（Key）',ylabel='当前预测位置（Query）'); save(fig,'causal-mask')
q=np.array([1.,0.]); k=np.array([[1.,0.],[0.,1.],[1.,1.]]); score=k@q/np.sqrt(2); probability=np.exp(score-score.max()); probability/=probability.sum(); fig,ax=plt.subplots(figsize=(6,3)); ax.bar(['位置1','位置2','位置3'],probability,color='#5486cc'); ax.set(title='一个 Query 对三个 Key 的注意力权重',ylabel='权重',ylim=(0,1)); save(fig,'attention')
flow('reversal',['输入 2 4 7','分隔符 SEP','生成 7 4 2','结束符 EOS'])
