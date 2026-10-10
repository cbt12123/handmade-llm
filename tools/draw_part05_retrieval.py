"""Plot source/formula checks, not general retrieval accuracy."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
cases = json.loads((ROOT/'data/part05/retrieval_cases.json').read_text(encoding='utf-8'))
before = json.loads((ROOT/'outputs/part05/24_retrieval_before.json').read_text(encoding='utf-8'))
after = json.loads((ROOT/'outputs/part05/24_retrieval_after.json').read_text(encoding='utf-8'))
old_hits = {row['query']:row['hits'] for row in before['queries']}
passed = 0
for case in cases:
    hits = old_hits[case['query']]
    if case.get('empty'):
        passed += not hits
    else:
        relevant = [h for h in hits if any(p in h['source'] for p in case['sources'])]
        passed += bool(relevant) and all(any(term in h['text'] for h in relevant) for term in case['contains'])
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
fig, ax = plt.subplots(figsize=(9, 5))
values = [passed, after['passed']]
bars = ax.bar(['旧段落索引', '公式与邻段索引'], values, color=['#94a3b8','#2563eb'], width=.5)
count = len(cases)
ax.set(ylim=(0,count+2), yticks=range(0,count+1,2), ylabel='满足指定来源与证据要求的用例数',
       title=f'{count} 条检索回归用例：同一组查询与要求')
for bar, value in zip(bars, values):
    ax.text(bar.get_x()+bar.get_width()/2,value+.25,f'{value} / {count}',ha='center',fontsize=14)
ax.spines[['top','right']].set_visible(False)
fig.text(.5,.03,'同时改变了知识范围、分块与排序；不能归因于单一因素，也不是通用准确率',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.06,1,1))
path = ROOT/'images/part05-retrieval-review.png'
fig.savefig(path,dpi=160)
plt.close(fig)
print(f'Before {passed}/{count}; after {after["passed"]}/{count}; {path}')
