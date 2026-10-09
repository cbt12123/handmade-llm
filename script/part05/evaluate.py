"""Actual model runs. Preserve failures; judge facts and actions separately."""
from pathlib import Path
import argparse
import hashlib
import json
import tempfile
import time
from common import ROOT, save
from knowledge import Knowledge
from model import HTTPModel
from runner import Agent
from store import Store
from toolbox import Toolbox
from presentation import present


def judge(case,state,store):
    kind = case['kind']
    done = state['status'] == 'done'
    if kind == 'explain':
        evidence_ok = bool(state['citations']) and all(c in state['evidence'] for c in state['citations'])
        return done and evidence_ok and all(t in state['answer'] for t in case['expected_terms'])
    if kind == 'quiz':
        rendered = present(state)
        return done and set(state['quizzes']) == {'gd-01','gd-02'} and all(q in rendered['answer'] for q in state['quizzes'])
    if kind in ('grade','record','denied'):
        grade = state['grades'].get(case['question_id'],{})
        good = done and grade.get('correct') is case['correct']
        count = len(store.recent(state['learner']))
        return good and (count == (1 if kind == 'record' else 0))
    if kind == 'unknown':
        return done and '资料不足' in state['answer'] and not state['citations']
    if kind == 'clarify':
        return done and not state['grades'] and '题号' in state['answer'] and '答案' in state['answer']
    if kind == 'boundary':
        # This measures capability isolation only, not correctness of the explanation.
        return all(e.get('action',{}).get('action') in ('load_skill','search','finish') for e in state['events'])
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8002')
    parser.add_argument('--model',default='tutorial-agent')
    args = parser.parse_args()
    dataset = ROOT/'data'/'part05'/'evaluation.jsonl'
    cases = [json.loads(line) for line in dataset.read_text(encoding='utf-8').splitlines() if line]
    model = HTTPModel(args.url,args.model)
    results = []
    with tempfile.TemporaryDirectory() as folder:
        store = Store(Path(folder)/'evaluation.sqlite3')
        try:
            agent = Agent(model,Toolbox(Knowledge(),store),store)
            for case in cases:
                start, offset = time.perf_counter(),len(model.calls)
                state = agent.run(agent.create(case['request'],case['id'],case.get('allow_record',False)))
                passed = judge(case,state,store)
                result = {'id':case['id'],'kind':case['kind'],'passed':passed,
                          'seconds':time.perf_counter()-start,'state':state,'presentation':present(state),'model_calls':model.calls[offset:]}
                results.append(result)
                print(case['id'],state['status'],'PASS' if passed else 'FAIL',flush=True)
                save('27_evaluation.json',{'backend':'vllm','model':args.model,
                     'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),
                     'count':len(results),'passed':sum(r['passed'] for r in results),'results':results,
                     'scope':'Eight authored cases, automated narrow checks. No general accuracy or semantic correctness guarantee.'})
        finally:
            store.close()
            model.session.close()


if __name__ == '__main__':
    main()
