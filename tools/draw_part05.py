"""Original Agent diagrams using matplotlib, no external image assets."""
from pathlib import Path
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                     'axes.unicode_minus':False,'font.size':11})


def box(ax,x,y,w,h,label,color='#e7efff'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.02',facecolor=color,edgecolor='#6386aa'))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center')


def arrow(ax,a,b):
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.8,'color':'#59636e'})


def canvas(title,width=11,height=4):
    fig,ax=plt.subplots(figsize=(width,height));ax.set(xlim=(0,12),ylim=(0,4));ax.axis('off')
    ax.text(6,3.65,title,ha='center',fontsize=15)
    return fig,ax


def save(fig,name):
    fig.tight_layout()
    fig.savefig(ROOT/'images'/f'part05-{name}.png',dpi=160)
    plt.close(fig)


fig,ax=canvas('Agent 循环：模型提出动作，代码检查并执行')
for x,label in [(0.2,'任务与当前状态'),(4.2,'模型\n选择下一动作'),(8.2,'参数、权限检查\n工具执行')]:
    box(ax,x,1.65,3.3,1,label)
arrow(ax,(3.55,2.15),(4.15,2.15));arrow(ax,(7.55,2.15),(8.15,2.15))
arrow(ax,(9.85,1.6),(9.85,.7));arrow(ax,(9.85,.7),(1.85,.7));arrow(ax,(1.85,.7),(1.85,1.6))
ax.text(6,.18,'结果回到状态；finish 通过检查则结束；最多八步',ha='center');save(fig,'loop')

fig,ax=canvas('Rules、Skills、Knowledge 分别解决不同问题')
for x,title,body,color in [(0.2,'Rules','行为与执行边界\n必须引用、写入授权','#ffeccc'),(4.2,'Skills','任务方法\n查询、练习、批改','#e7efff'),(8.2,'Knowledge','任务资料\n教程、题库与来源','#e4f4eb')]:
    box(ax,x,1.45,3.3,1.5,title+'\n\n'+body,color)
box(ax,3.4,.15,5.2,.7,'Python 工具落实检索、评分和保存');save(fig,'components')

fig,ax=canvas('工具调用的检查层次',height=3.6)
labels=['JSON 字段\n能否解析','固定工具名\n是否登记','技能权限\n是否允许','参数与写权限\n是否满足','执行与提交\n是否成功']
for i,label in enumerate(labels):
    box(ax,.12+i*2.4,1.3,2.05,1.1,label)
    if i<4:arrow(ax,(2.21+i*2.4,1.85),(2.48+i*2.4,1.85))
ax.text(6,.55,'任一层失败：不视为成功，记录错误或结束任务',ha='center');save(fig,'boundaries')

fig,ax=canvas('技能按需加载，代码按目录检查权限')
box(ax,.3,1.7,3.1,1,'catalog\n名称、描述、允许工具')
box(ax,4.45,1.7,3.1,1,'load_skill\n模型选择已登记技能')
box(ax,8.6,1.7,3.1,1,'SKILL.md\n加载当前任务方法')
arrow(ax,(3.45,2.2),(4.4,2.2));arrow(ax,(7.6,2.2),(8.55,2.2))
ax.text(6,.65,'说明文件指导工作；工具代码实际执行；附件不会自行运行',ha='center');save(fig,'skills')

fig,ax=canvas('知识索引与 RAG：资料是数据，回答可回查')
for i,label in enumerate(['教程正文\n路径与行号','分块索引\n内容 ID、哈希','查询与排序\n返回两个块','模型生成\n回答与引用']):
    box(ax,.2+i*3,1.5,2.6,1.2,label)
    if i<3:arrow(ax,(2.85+i*3,2.1),(3.15+i*3,2.1))
ax.text(6,.65,'正文变化后重建；真实引用不自动保证解释正确',ha='center');save(fig,'knowledge')

fig,ax=plt.subplots(figsize=(8,5));
for name,v,color in [('q',(1,2),'#0969da'),('A',(2,4),'#2da44e'),('B',(2,0),'#bf8700'),('C',(-1,-2),'#cf222e')]:
    ax.annotate('',xy=v,xytext=(0,0),arrowprops={'arrowstyle':'->','color':color,'lw':2})
    ax.text(v[0]+.08,v[1]+.08,name+' '+str(v),color=color)
ax.axhline(0,color='#ccc');ax.axvline(0,color='#ccc');ax.set(xlim=(-2,3.5),ylim=(-3,5),xlabel='第一维',ylabel='第二维',title='手工向量：方向相同 1，反向 -1');ax.grid(alpha=.2);save(fig,'vectors')

fig,ax=canvas('状态与记忆：数据库里有记录，不等于模型已经读取')
box(ax,.3,1.45,3.1,1.35,'当前上下文\n原目标、当前技能\n近期结果')
box(ax,4.45,1.45,3.1,1.35,'任务检查点\ntasks 表\n恢复当前任务','#e4f4eb')
box(ax,8.6,1.45,3.1,1.35,'长期学习记录\nrecords 表\nprogress 工具读取','#ffeccc')
arrow(ax,(3.45,2.1),(4.4,2.1));arrow(ax,(8.55,1.75),(3.45,1.1));ax.text(6,.45,'记录写入与检查点在同一事务中提交',ha='center');save(fig,'memory')

fig,ax=canvas('完整学习助手的运行关系')
box(ax,.25,1.6,2.6,1.2,'CLI / HTTP\n用户与授权')
box(ax,3.5,1.6,2.6,1.2,'Python Agent\n循环、工具与状态')
box(ax,7.1,2.15,4.4,.9,'vLLM 容器\n模型选择动作')
box(ax,7.1,.75,4.4,.9,'本地文件 + SQLite\n知识、技能、记录','#e4f4eb')
arrow(ax,(2.9,2.2),(3.45,2.2));arrow(ax,(6.15,2.4),(7.05,2.6));arrow(ax,(6.15,1.8),(7.05,1.2));save(fig,'architecture')

report=ROOT/'outputs'/'part05'/'27_evaluation.json'
if report.exists():
    data=json.loads(report.read_text(encoding='utf-8'))
    fig,ax=plt.subplots(figsize=(10,4.5))
    ax.bar(range(len(data['results'])),[r['passed'] for r in data['results']],color=['#2da44e' if r['passed'] else '#cf222e' for r in data['results']])
    ax.set(xticks=range(len(data['results'])),xticklabels=[r['id'] for r in data['results']],yticks=[0,1],yticklabels=['未通过','通过'],ylim=(0,1.2),title='实际模型的八条教学用例：各项检查范围不同')
    ax.tick_params(axis='x',rotation=25);save(fig,'evaluation')
