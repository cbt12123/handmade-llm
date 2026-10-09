"""Original diagrams and numerical plots for Part IV; use base or handmade-ml."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "images"
plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],
                     "axes.unicode_minus":False,"font.size":11})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT/f"part04-{name}.png",dpi=160)
    plt.close(fig)


def block(ax, x, y, width, height, text, color="#e7efff"):
    ax.add_patch(FancyBboxPatch((x,y),width,height,boxstyle="round,pad=.03",
                              facecolor=color,edgecolor="#6386aa"))
    ax.text(x+width/2,y+height/2,text,ha="center",va="center")


def arrow(ax, start, end):
    ax.annotate("",xy=end,xytext=start,arrowprops={"arrowstyle":"->","color":"#59636e","lw":1.7})


fig, ax = plt.subplots(figsize=(11,3.4)); ax.axis("off"); ax.set(xlim=(0,12),ylim=(0,3))
for i,label in enumerate(["消息与模板","分词\nID序列","嵌入与\n28层Transformer","词表logits","选择下一ID"]):
    block(ax,.1+i*2.4,1,2.05,.95,label)
    if i<4: arrow(ax,(2.22+i*2.4,1.48),(2.43+i*2.4,1.48))
ax.text(6,2.5,"语言模型推理：固定权重，逐token生成",ha="center",fontsize=15)
arrow(ax,(10.7,.95),(10.7,.35)); arrow(ax,(10.7,.35),(3.55,.35)); arrow(ax,(3.55,.35),(3.55,.95))
ax.text(7,.02,"新ID加入前缀，继续预测",ha="center")
save(fig,"model-flow")

fig,ax=plt.subplots(figsize=(10,4));ax.axis("off");ax.set(xlim=(0,12),ylim=(0,4))
for i in range(12): block(ax,.15+i*.98,2.6,.8,.6,f"Q{i+1}")
block(ax,1.3,.7,3.1,.85,"K1、V1：128维", "#e5f4eb")
block(ax,7.2,.7,3.1,.85,"K2、V2：128维", "#fff0d9")
for i in range(12): arrow(ax,(i*.98+.55,2.55),(2.85 if i<6 else 8.75,1.6))
ax.text(6,3.6,"12个Query头分为两组，每组共享一组KV",ha="center",fontsize=14)
ax.text(6,.08,"KV缓存保存2个头的历史Key和Value，不保存12份重复数据",ha="center")
save(fig,"gqa")

fig,ax=plt.subplots(figsize=(10,4));ax.axis("off");ax.set(xlim=(0,10),ylim=(0,4))
for x,y,label in [(0.2,2.2,"config.json\n层数、维度、模型类型"),(5.2,2.2,"model.safetensors\n已训练参数"),(.2,.5,"tokenizer文件\n词表、编码、对话模板"),(5.2,.5,"generation_config.json\n部分生成默认值")]:
    block(ax,x,y,4.3,1.2,label)
ax.text(5,3.65,"完整模型目录：结构、权重与输入规则保持配套",ha="center",fontsize=14)
save(fig,"model-files")

z=np.array([2.,1.,0.,-1.]); fig,ax=plt.subplots(figsize=(8,4))
for offset,temp in [(-.18,1),(.18,.5)]:
    p=np.exp((z-z.max())/temp);p/=p.sum()
    ax.bar(np.arange(4)+offset,p,width=.36,label=f"温度 {temp}")
ax.set(xticks=range(4),xlabel="候选下标",ylabel="概率",ylim=(0,1),title="相同logits，不同温度");ax.legend();save(fig,"sampling")

fig,ax=plt.subplots(figsize=(11,4));ax.axis("off");ax.set(xlim=(0,11),ylim=(0,4))
block(ax,.15,2.25,3.2,.85,"首次prefill\n完整提示：T个token")
block(ax,4,2.25,2.8,.85,"各层Key、Value\n长度T", "#e5f4eb")
block(ax,7.5,2.25,3.2,.85,"第一个新增token\n由最后位置logits选择")
arrow(ax,(3.45,2.65),(3.9,2.65));arrow(ax,(6.9,2.65),(7.4,2.65))
block(ax,.15,.55,3.2,.85,"下一次decode\n只输入上一步新token")
block(ax,4,.55,2.8,.85,"读取旧KV，追加新KV\n长度T+1", "#e5f4eb")
block(ax,7.5,.55,3.2,.85,"第二个新增token\n继续同样过程")
arrow(ax,(3.45,.95),(3.9,.95));arrow(ax,(6.9,.95),(7.4,.95));arrow(ax,(5.4,2.2),(5.4,1.45))
ax.text(5.5,3.6,"KV Cache复用历史中间结果，不保存一份现成答案",ha="center",fontsize=14)
save(fig,"cache")

fig,axes=plt.subplots(1,2,figsize=(10,4))
axes[0].bar(["FP32","FP16","INT8*","INT4*"],np.array([4,2,1,.5])*1.54,color="#7399c8")
axes[0].set(ylabel="权重理想数据量 / GB",title="约1.54B参数；*不含量化元数据")
lengths=np.array([256,1024,4096,8192]);per_token=2*28*2*128*2
for batch in [1,4]: axes[1].plot(lengths,lengths*per_token*batch/2**20,"o-",label=f"序列数 {batch}")
axes[1].set(xlabel="缓存长度",ylabel="FP16 KV / MiB",title="权重之外还需缓存与工作区");axes[1].legend();save(fig,"memory")

rng=np.random.default_rng(42);w=rng.normal(size=(8,16)).astype(np.float32);w[0,0]=40
fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,per_row,title in zip(axes,[False,True],["全张量scale","逐输出行scale"]):
    maximum=np.max(np.abs(w),axis=1,keepdims=True) if per_row else np.max(np.abs(w))
    scale=maximum/127;recovered=np.clip(np.rint(w/scale),-127,127).astype(np.int8).astype(np.float32)*scale
    picture=ax.imshow(np.abs(w-recovered),vmin=0,vmax=.16,cmap="Oranges")
    ax.set(title=title,xlabel="输入维",ylabel="输出行")
fig.colorbar(picture,ax=axes.ravel().tolist(),label="权重绝对误差",shrink=.8)
fig.subplots_adjust(left=.08,right=.82,wspace=.35,top=.85,bottom=.16)
fig.savefig(OUT/"part04-quantization.png",dpi=160);plt.close(fig)

fig,ax=plt.subplots(figsize=(11,4));ax.axis("off");ax.set(xlim=(0,11),ylim=(0,4))
for x,label in [(0.1,"HTTP客户端\n消息与输出限制"),(3.7,"FastAPI\n校验、预算、并发限制"),(7.3,"常驻模型\n模板、生成、解码")]:block(ax,x,1.5,3.1,1.1,label)
arrow(ax,(3.3,2.05),(3.6,2.05));arrow(ax,(6.9,2.05),(7.2,2.05))
ax.text(5.5,3.3,"启动时加载一次；请求时使用同一模型实例",ha="center",fontsize=14)
ax.text(5.5,.65,"200：成功   422：请求字段错误   413：token预算超限   429：模型忙",ha="center")
save(fig,"service")

report=ROOT/"outputs/part04/19_evaluation.json"
if report.exists():
    data=json.loads(report.read_text(encoding="utf-8"));cats=data["categories"]
    names=list(cats);labels={"arithmetic":"整数计算","json":"严格JSON提取","missing_information":"缺少资料"}
    fig,ax=plt.subplots(figsize=(8,4))
    ax.bar([labels[n] for n in names],[cats[n]["correct"]/cats[n]["count"] for n in names],color="#7399c8")
    for i,n in enumerate(names): ax.text(i,.05,f"{cats[n]['correct']}/{cats[n]['count']}",ha="center")
    ax.set(ylim=(0,1.1),ylabel="本次用例正确比例",title="仅8条自编教学用例，不是通用能力排名")
    save(fig,"evaluation")
