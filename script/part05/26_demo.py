"""Real complete learning workflow in an isolated temporary database."""
from pathlib import Path
import json
import tempfile
from common import save
from knowledge import Knowledge
from model import HTTPModel
from presentation import present
from runner import Agent
from store import Store
from toolbox import Toolbox


def main():
    model = HTTPModel()
    runs = []
    with tempfile.TemporaryDirectory() as folder:
        store = Store(Path(folder)/'demo.sqlite3')
        try:
            agent = Agent(model,Toolbox(Knowledge(),store),store)
            requests = [
                ('explain','请根据教程解释梯度下降，给出出处。',False,['search']),
                ('practice','请给我两道梯度下降练习，先不要给答案。',False,['quiz']),
                ('review','gd-01 的答案是 2，请检查并保存复习记录。',True,['grade','record']),
                ('progress','请使用 review 技能的 progress 工具查询我的最近学习记录。',False,['progress']),
            ]
            for label,request,record,required in requests:
                state = agent.run(agent.create(request,'demo-fixture',record,required))
                result = present(state)
                runs.append({'label':label,'state':state,'presentation':result})
                print(label,state['status'],result['answer'],flush=True)
            checks = {
                'all_tasks_finished':all(r['state']['status']=='done' for r in runs),
                'explanation_has_source':bool(runs[0]['state']['citations']),
                'two_questions':set(runs[1]['state']['quizzes'])=={'gd-01','gd-02'},
                'wrong_answer_scored_and_saved':runs[2]['state']['grades'].get('gd-01',{}).get('correct') is False
                    and runs[2]['state']['recorded']==['gd-01'],
                'progress_tool_called':any(e.get('action',{}).get('action')=='progress' and 'observation' in e for e in runs[3]['state']['events']),
                'record_persisted_across_tasks':len(store.recent('demo-fixture'))==1,
            }
            save('26_demo.json',{'scope':'Real model-backed fixed four-task workflow; temporary database; prose quality requires separate review',
                                 'checks':checks,'runs':runs,'model_calls':model.calls})
            print(json.dumps(checks,indent=2))
            if not all(checks.values()):
                raise SystemExit(1)
        finally:
            store.close()
            model.session.close()


if __name__=='__main__':
    main()
