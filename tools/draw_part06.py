"""Original operator diagrams and plots from measured reports, CPU-only."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                    'axes.unicode_minus':False,'font.size':11})


def read(name):return json.loads((ROOT/'outputs/part06'/name).read_text(encoding='utf-8'))
def save(fig,name):
    fig.tight_layout();fig.savefig(ROOT/'images'/f'part06-{name}.png',dpi=160);plt.close(fig)
def canvas(title):
    fig,ax=plt.subplots(figsize=(11,4));ax.set(xlim=(0,12),ylim=(0,4));ax.axis('off')
    ax.text(6,3.65,title,ha='center',fontsize=15);return fig,ax
def box(ax,x,y,w,h,label,color='#e7efff'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.025',facecolor=color,edgecolor='#6386aa'))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center')
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.6,'color':'#59636e'})
def chain(name,title,labels,foot):
    fig,ax=canvas(title);w=10.8/len(labels)-.35
    for i,label in enumerate(labels):
        x=.3+i*(w+.35);box(ax,x,1.55,w,1.3,label)
        if i:arrow(ax,(x-.3,2.2),(x-.05,2.2))
    ax.text(6,.65,foot,ha='center');save(fig,name)


def main():
    chain('evidence','算子交付需要四类证据',['数值正确\n边界与精度','真实调用\n模型与Profiler','整体收益\n一致条件对照','可复现交付\n环境与回退'],
          '独立函数能跑，尚不足以证明真实推理得到改善')
    chain('stack','从请求到GPU执行',['HTTP请求\n提示词与预算','生成循环\n缓存与选择','模型模块\n57处RMSNorm','GPU内核\n融合归一化'],
          '本次实际路径：Transformers + SDPA + Triton；使用已有镜像中的库')
    fig,ax=canvas('全局索引与尾部：n=1003，每块256线程')
    for i in range(4):box(ax,.3+i*2.9,1.7,2.6,1.05,f'block {i}\n索引 {i*256}～{i*256+255}')
    ax.text(6,.9,'前三块全部有效；最后一块仅768～1002有效，1003～1023不访问数组',ha='center')
    ax.text(6,.3,'i = blockIdx.x × blockDim.x + threadIdx.x；访问前判断 i < n',ha='center');save(fig,'index')
    fig,ax=canvas('分层归约：部分和 → 最终和 → 广播')
    for i,label in enumerate(['warp 0部分和','warp 1部分和','…','warp 7部分和']):box(ax,.3+i*2.9,2.25,2.6,.6,label)
    box(ax,3.5,1.2,5,.65,'warp 0汇总 → 计算共享 inverse',color='#e4f4eb')
    for i in range(4):arrow(ax,(1.6+i*2.9,2.2),(6,1.9))
    box(ax,2.1,.15,7.8,.6,'同步后，所有warp读取同一个 inverse，计算各自输出')
    arrow(ax,(6,1.15),(6,.8));save(fig,'reduction')
    chain('tiles','分块矩阵乘法：复用前后都要同步',['读入A与B子块\n保护边界','同步\n等待装载','复用共享数据\n累加局部结果','同步\n再覆盖子块'],
          '学习数据复用；本次教学实现仍慢于cuBLAS，不替代成熟GEMM')
    chain('fusion','RMSNorm：把中间步骤放进一次融合计算',['读入一行\n与权重','FP32归约\n平方和 / D','计算归一化\n先转换输入精度','乘权重\n写回输出'],
          'Qwen约定：归一化后先转换回输入类型，再乘权重；不能随意省掉舍入')
    data=read('31_microbenchmark.json')['results'];fig,ax=plt.subplots(figsize=(10,4.6))
    names=['torch_eager','torch_compile','triton_4_warps','triton_8_warps']
    actual=list(data[0]['measurements']);names=actual
    x=np.arange(len(data));w=.18
    for i,name in enumerate(names):ax.bar(x+(i-1.5)*w,[r['measurements'][name]['wall_median_ms'] for r in data],w,label=name)
    ax.set(xticks=x,xticklabels=[str(r['shape'][0]) for r in data],xlabel='行数（宽度1536，FP16）',ylabel='墙钟中位数 / ms',title='同一归一化函数：eager、compile与自写Triton')
    ax.legend(fontsize=9);ax.text(.5,-.23,'实际多调用微基准；含包装与提交成本，不是纯内核时间',ha='center',transform=ax.transAxes);save(fig,'microbenchmark')
    chain('selection','根据需要的输出减少计算',['输出需求\n下一token','最后隐藏位置\n词表投影','直接argmax\n按logits选择','得到token\n继续缓存生成'],
          '需要整段logits或概率采样时，不能直接套用这些省略')
    profile=read('33_profile.json')['results'];fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key,title in zip(axes,['cuda_event_count','course_rms_kernel_count'],['记录到的CUDA事件','自写_rms内核事件']):
        vals=[profile[n][key] for n in ['baseline','fused']];ax.bar(['baseline','fused'],vals,color=['#8da8cc','#53a68c'])
        ax.set(title=title,ylabel='次数')
        for i,v in enumerate(vals):ax.text(i,v+max(vals)*.025,str(v),ha='center')
        ax.set_ylim(0,max(vals)*1.15)
    fig.suptitle('真实模型生成4步：57 × 4 = 228；Profiler不用于性能基准');save(fig,'profile')
    chain('inference','适配器暂时改变真实模型执行路径',['保存57处\n原forward','支持布局\n调用Triton','记录生成\n比较完整IDs','finally恢复\n包括异常退出'],
          '不支持的输入调用原实现；本次实际GPU测试回退为零')
    rows=read('34_real_inference.json')['measurements'];fig,ax=plt.subplots(figsize=(10,4.8));x=np.arange(len(rows))
    for i,(key,label) in enumerate([('total_median_speedup','生成32步'),('prefill_median_speedup','单独预填充')]):
        vals=[r[key] for r in rows];ax.bar(x+(i-.5)*.3,vals,.3,label=label)
        for j,v in enumerate(vals):ax.text(j+(i-.5)*.3,v+.025,f'{v:.3f}',ha='center')
    ax.axhline(1,color='#777',linestyle='--');ax.set(xticks=x,xticklabels=[str(r['input_tokens']) for r in rows],ylim=(0,1.4),xlabel='输入token数',ylabel='baseline耗时 / fused耗时',title='真实Qwen生成：本次中位数的加速比（越高越快）')
    ax.legend();ax.text(.5,-.22,'相同FP16 / SDPA / KV缓存 / argmax / 最后位置logits；1024预填充略慢',ha='center',transform=ax.transAxes);save(fig,'results')
    chain('deployment','部署后的请求经过真实融合算子',['Windows客户端\nHTTP :8003','串行入口\n预算与锁','真实Qwen\n默认fused','返回IDs与计数\n可选baseline'],
          '服务遇EOS停止；三条真实提示词的完整IDs与调用计数均已检查')
    print('Generated 12 original figures')


if __name__=='__main__':main()
