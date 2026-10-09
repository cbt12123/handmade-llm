"""Original mathematical illustrations for part 2; base NumPy + Matplotlib."""
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "images"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "script" / "part02"))
from _common import transform_figure, graph_figure
plt.rcParams.update({"font.size":11, "axes.unicode_minus":False})
available = {font.name for font in font_manager.fontManager.ttflist}
plt.rcParams["font.sans-serif"] = [name for name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"] if name in available]
TRANSLATIONS = {
    "Average change approaches local change":"平均变化率趋近局部变化率",
    "Forward: compute values; backward: multiply derivatives (6 x 2 = 12)":"前向计算数值；反向逐段相乘导数（6 × 2 = 12）",
    "One update: (0,0) -> (0.6,-0.2)":"一次更新：(0,0) → (0.6,-0.2)",
    "Minimum (3,-1)":"最小值位置 (3,-1)",
    "Learning rate changes convergence":"学习率改变收敛过程",
    "Loss (floor 1e-15)":"损失（显示下限 1e-15）",
    "Update step":"更新步数",
    "Midpoint rectangles: sum of speed x time":"中点矩形近似：累加速度 × 时长",
    "Time":"时间", "Speed":"速度",
    "Vector addition":"向量加法",
    "Original":"原图形", "Scale":"缩放", "Rotate 45 degrees":"旋转 45 度", "Project to x-axis":"投影到 x 轴",
    "Least squares with noisy observations":"带噪声观察值的最小二乘拟合",
    "Observations":"观察值",
    "Frequency fluctuates, even for a fair coin":"公平硬币的有限次频率也会波动",
    "Trials":"实验次数", "Heads frequency":"正面频率",
    "Among 585 positives, only 90 are defective":"585 个预期阳性样品中，90 个有缺陷",
    "Expected count among 10,000 items":"10000 个样品中的预期数量",
    "Defective: true positive":"有缺陷：正确检出", "Not defective: false positive":"无缺陷：误报",
    "Fair die: discrete probabilities":"公平骰子：各结果的离散概率",
    "Outcome":"结果", "Probability":"概率",
    "Normal: interval probability is area":"正态密度：区间概率是面积",
    "Value":"数值", "Density":"密度",
    "Means of independent samples":"独立样本的均值分布",
    "Sample mean":"样本均值",
    "Cross entropy for one true class: -ln(p)":"真实标签固定时的交叉熵：-ln(p)",
    "Probability assigned to true class":"分配给真实类别的概率", "Loss":"损失",
    "Intersection: {3}":"交集：{3}", "Union: {1,2,3,4}":"并集：{1,2,3,4}", "A minus B: {1,2}":"差集 A-B：{1,2}",
    "Causal: j <= i":"因果可见：j ≤ i", "Valid key positions":"有效的被关注位置", "Combined (AND)":"合并条件：同时成立",
    "Key position j":"被关注位置 j", "Query position i":"查询位置 i",
    "Weighted DAG: A -> B -> C -> D costs 4":"带权 DAG：A → B → C → D 成本为 4",
    "Growth of work, not elapsed seconds":"工作量的增长趋势（不是运行秒数）",
    "Input size n":"输入规模 n", "Operation-count scale (log)":"操作数量尺度（对数轴）",
}

def save(fig, name):
    for item in fig.findobj(match=Text):
        value = item.get_text()
        item.set_text(TRANSLATIONS.get(value, value))
    fig.tight_layout()
    fig.savefig(OUT / f"part02-{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(10,7))
x=np.linspace(-2,2,200)
for ax,y,title in zip(axes.flat,[2*x+1,x*x,np.exp(x),np.log(np.linspace(.1,4,200))],["f(x)=2x+1","f(x)=x^2","f(x)=exp(x)","f(x)=ln(x), x>0"]):
    xx=np.linspace(.1,4,200) if "ln" in title else x
    ax.plot(xx,y); ax.set(title=title,xlabel="x",ylabel="f(x)"); ax.grid(alpha=.25)
save(fig,"functions")

fig,ax=plt.subplots(figsize=(8,4.5))
x=np.linspace(.5,3.5,200)
ax.plot(x,x*x,label="f(x)=x^2")
ax.plot(x,4+4*(x-2),"--",label="Tangent at x=2: slope 4")
for h in [1,.3]:
    xx=np.array([2,2+h]); ax.plot(xx,xx*xx,"o-",label=f"Secant h={h}: slope {4+h}")
ax.set(xlabel="x",ylabel="f(x)",title="Average change approaches local change")
ax.legend(); ax.grid(alpha=.2)
save(fig,"derivative")

fig,ax=plt.subplots(figsize=(7,5))
w=np.linspace(-1,5,100); b=np.linspace(-3,2,100)
ww,bb=np.meshgrid(w,b)
contours=ax.contour(ww,bb,(ww-3)**2+(bb+1)**2,levels=[.5,2,5,10,16,25])
ax.clabel(contours)
ax.plot(3,-1,"*",markersize=15,label="Minimum (3,-1)")
ax.quiver(0,0,.6,-.2,angles="xy",scale_units="xy",scale=1,color="#d97732")
ax.plot(0,0,"o"); ax.annotate("One update: (0,0) -> (0.6,-0.2)",xy=(.6,-.2),xytext=(-.5,1.2),arrowprops=dict(arrowstyle="->"))
ax.set(xlabel="w",ylabel="b",title="L(w,b)=(w-3)^2+(b+1)^2"); ax.legend()
save(fig,"gradient")

fig,ax=plt.subplots(figsize=(10,3.5)); ax.set(xlim=(0,10),ylim=(0,3)); ax.axis("off")
for x,label in [(0.3,"x=1"),(3.8,"u=2x+1\nu=3"),(7.3,"L=u^2\nL=9")]:
    ax.add_patch(FancyBboxPatch((x,1),2.3,1,boxstyle="round,pad=.1",facecolor="#e7f0ff",edgecolor="#5f7899"))
    ax.text(x+1.15,1.5,label,ha="center",va="center",fontsize=14)
for start,end in [(2.7,3.6),(6.2,7.1)]:
    ax.annotate("",xy=(end,1.7),xytext=(start,1.7),arrowprops=dict(arrowstyle="->"))
    ax.annotate("",xy=(start,.65),xytext=(end,.65),arrowprops=dict(arrowstyle="->",color="#d97732"))
ax.text(3.15,.2,"du/dx=2",ha="center"); ax.text(6.65,.2,"dL/du=6",ha="center")
ax.set_title("Forward: compute values; backward: multiply derivatives (6 x 2 = 12)")
save(fig,"chain")

fig,ax=plt.subplots(figsize=(8,4.5))
for lr in [.05,.4,1.1]:
    w=0.; values=[9.]
    for _ in range(30):
        w-=lr*2*(w-3); values.append((w-3)**2)
    ax.semilogy(np.maximum(values,1e-15),label=f"lr={lr}")
ax.set(title="Learning rate changes convergence",xlabel="Update step",ylabel="Loss (floor 1e-15)"); ax.legend(); ax.grid(alpha=.2)
save(fig,"learning-rate")

fig,ax=plt.subplots(figsize=(8,4.5))
x=np.linspace(0,2,100); ax.plot(x,2*x,label="v(t)=2t")
width=.25
for left in np.arange(0,2,width):
    ax.add_patch(Rectangle((left,0),width,2*(left+width/2),facecolor="#bbd4f6",edgecolor="white",alpha=.7))
ax.set(title="Midpoint rectangles: sum of speed x time",xlabel="Time",ylabel="Speed",ylim=(0,4.5)); ax.legend()
save(fig,"integral")

fig,axes=plt.subplots(1,2,figsize=(10,4))
ax=axes[0]
for start,end,color,label in [((0,0),(2,1),"#2863a1","u=(2,1)"),((2,1),(3,3),"#dc7b36","v=(1,2)"),((0,0),(3,3),"#29976e","u+v=(3,3)")]:
    ax.annotate("",xy=end,xytext=start,arrowprops=dict(arrowstyle="->",color=color,lw=2))
    label_y = end[1]-.45 if label.startswith("v=") else end[1]+.15
    ax.text(end[0]-.4,label_y,label,color=color)
ax.set(xlim=(-.5,4),ylim=(-.5,4),title="Vector addition"); ax.set_aspect("equal"); ax.grid(alpha=.2)
ax=axes[1]; ax.plot([0,2,2],[0,0,1],"o--"); ax.annotate("",xy=(2,1),xytext=(0,0),arrowprops=dict(arrowstyle="->",lw=2))
ax.set(xlim=(-.5,3),ylim=(-.5,2),title="(2,1) = 2 e1 + 1 e2"); ax.set_aspect("equal"); ax.grid(alpha=.2)
save(fig,"vectors")
save(transform_figure(),"transforms")

fig,ax=plt.subplots(figsize=(7,4.5))
x=np.array([0.,1.,2.,3.]); y=np.array([1.1,2.9,5.2,6.8])
design=np.column_stack([x,np.ones_like(x)]); wb=np.linalg.lstsq(design,y,rcond=None)[0]
ax.scatter(x,y,label="Observations"); ax.plot(x,design@wb,label=f"Fit: y={wb[0]:.2f}x+{wb[1]:.2f}")
ax.set(xlabel="x",ylabel="y",title="Least squares with noisy observations"); ax.legend()
save(fig,"least-squares")

rng=np.random.default_rng(42); heads=rng.integers(0,2,10000); frequency=np.cumsum(heads)/np.arange(1,10001)
fig,ax=plt.subplots(figsize=(8,4.5)); ax.plot(np.arange(1,10001),frequency); ax.axhline(.5,color="black",linestyle="--")
ax.set(title="Frequency fluctuates, even for a fair coin",xlabel="Trials",ylabel="Heads frequency",ylim=(0,1))
save(fig,"frequency")

fig,ax=plt.subplots(figsize=(7,4.5)); ax.bar(["Defective: true positive","Not defective: false positive"],[90,495],color=["#e29448","#669bcc"])
for i,v in enumerate([90,495]): ax.text(i,v+8,str(v),ha="center")
ax.set(title="Among 585 positives, only 90 are defective",ylabel="Expected count among 10,000 items",ylim=(0,560))
save(fig,"bayes")

fig,axes=plt.subplots(1,2,figsize=(10,4))
axes[0].bar(np.arange(1,7),np.full(6,1/6)); axes[0].set(title="Fair die: discrete probabilities",xlabel="Outcome",ylabel="Probability")
x=np.linspace(-4,4,300); density=np.exp(-x*x/2)/np.sqrt(2*np.pi)
axes[1].plot(x,density); axes[1].fill_between(x,density,where=(x>=-1)&(x<=1),alpha=.3)
axes[1].set(title="Normal: interval probability is area",xlabel="Value",ylabel="Density")
save(fig,"distributions")

rng=np.random.default_rng(10)
fig,ax=plt.subplots(figsize=(8,4.5))
for n in [10,100]: ax.hist(rng.normal(size=(1000,n)).mean(axis=1),bins=35,alpha=.5,label=f"n={n}",density=True)
ax.set(title="Means of independent samples",xlabel="Sample mean",ylabel="Density"); ax.legend()
save(fig,"sampling")

fig,ax=plt.subplots(figsize=(7,4.5)); p=np.linspace(.01,1,300); ax.plot(p,-np.log(p)); ax.scatter([.1,.9],-np.log([.1,.9]))
ax.set(title="Cross entropy for one true class: -ln(p)",xlabel="Probability assigned to true class",ylabel="Loss"); ax.grid(alpha=.2)
save(fig,"cross-entropy")

fig,axes=plt.subplots(1,3,figsize=(11,3.5))
for ax,title,selected in zip(axes,["Intersection: {3}","Union: {1,2,3,4}","A minus B: {1,2}"],[{"3"},{"1","2","3","4"},{"1","2"}]):
    ax.add_patch(Circle((.42,.5),.29,color="#6299d4",alpha=.4)); ax.add_patch(Circle((.69,.5),.29,color="#e39853",alpha=.4))
    for x,y,label in [(.29,.57,"1"),(.29,.42,"2"),(.55,.5,"3"),(.82,.5,"4")]:
        ax.text(x,y,label,ha="center",va="center",fontsize=15,weight="bold" if label in selected else "normal",bbox=dict(boxstyle="circle",facecolor="#c6f3d9",edgecolor="#278857") if label in selected else None)
    ax.text(.25,.86,"A"); ax.text(.8,.86,"B"); ax.set(xlim=(0,1.1),ylim=(0,1),title=title); ax.set_aspect("equal"); ax.axis("off")
save(fig,"sets")

fig,axes=plt.subplots(1,3,figsize=(11,3.7)); causal=np.tril(np.ones((4,4),dtype=bool)); valid=np.broadcast_to(np.array([1,1,1,0],dtype=bool),(4,4))
for ax,data,title in zip(axes,[causal,valid,causal&valid],["Causal: j <= i","Valid key positions","Combined (AND)"]):
    ax.imshow(data,vmin=0,vmax=1,cmap="Blues")
    for i in range(4):
        for j in range(4): ax.text(j,i,int(data[i,j]),ha="center",va="center",color="white" if data[i,j] else "black")
    ax.set(title=title,xlabel="Key position j",ylabel="Query position i",xticks=range(4),yticks=range(4))
save(fig,"mask")
save(graph_figure(),"graph")

fig,ax=plt.subplots(figsize=(8,4.5)); n=np.arange(1,21)
for y,label in [(n,"n"),(n*n,"n^2"),(2**n,"2^n")]: ax.semilogy(n,y,label=label)
ax.set(title="Growth of work, not elapsed seconds",xlabel="Input size n",ylabel="Operation-count scale (log)"); ax.legend(); ax.grid(alpha=.2)
save(fig,"complexity")
print("Generated 18 mathematical teaching images.")
