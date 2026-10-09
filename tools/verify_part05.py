"""Meaningful boundary and recovery tests; scripted actions are explicit fixtures."""
from pathlib import Path
import json
import math
import sys
import tempfile
from contextlib import ExitStack

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'script'/'part05'))
from common import save
from knowledge import Knowledge, build_index
from model import validate_action
from runner import Agent
from store import Store
from toolbox import Toolbox
from presentation import present


def action(name, args=None, answer='', citations=None):
    return {'action':name,'args':args or {},'answer':answer,'citations':citations or []}


class ScriptedModel:
    """A test double for engine behavior, never counted as model performance."""
    def __init__(self, actions):
        self.actions = iter(actions)

    def decide(self,state):
        return next(self.actions)


def main():
    index = build_index()
    knowledge = Knowledge()
    checks = {}
    hits = knowledge.search('梯度下降')
    assert hits and any('梯度下降' in h['heading']+h['text'] for h in hits)
    assert knowledge.search('zzzzqnonexistent') == []
    checks['lexical_search_and_no_match'] = True
    # Verify sources and line references resolve; changing a hash invalidates loading.
    for h in hits:
        assert (ROOT/h['source']).is_file() and h['line'] >= 1
    with tempfile.TemporaryDirectory() as folder, ExitStack() as cleanup:
        base = Path(folder)
        changed = json.loads(json.dumps(index))
        changed['sources'][next(iter(changed['sources']))] = 'bad-hash'
        bad_path = base/'index.json'
        bad_path.write_text(json.dumps(changed),encoding='utf-8')
        try:
            Knowledge(bad_path)
        except ValueError:
            checks['stale_index_rejected'] = True
        else:
            raise AssertionError('Stale index accepted')
        store = Store(base/'test.sqlite3')
        cleanup.callback(store.close)
        tools = Toolbox(knowledge,store)
        agent = Agent(ScriptedModel([]),tools,store)
        state = agent.create('gd-01: 1.6 或 2')
        for name,args in [('shell',{'command':'del'}),('search',{'query':'梯度下降'})]:
            try:
                tools.execute(name,args,state)
            except ValueError:
                pass
            else:
                raise AssertionError('Ungated tool accepted')
        checks['registry_and_skill_gate'] = True
        tools.execute('load_skill',{'name':'review'},state)
        for bad in ['NaN','inf','1+1','', '1e9999']:
            try:
                tools.execute('grade',{'question_id':'gd-01','answer':bad},state)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid numeric answer accepted')
        assert tools.execute('grade',{'question_id':'gd-01','answer':'1.6'},state)['correct']
        try:
            tools.execute('grade',{'question_id':'gd-01','answer':'2'},state)
        except ValueError:
            checks['cannot_change_submitted_answer'] = True
        else:
            raise AssertionError('Changed answer accepted')
        state['grades'].clear()
        assert not tools.execute('grade',{'question_id':'gd-01','answer':'2'},state)['correct']
        checks['numeric_grading'] = True
        absent = agent.create('请批改我的练习')
        tools.execute('load_skill',{'name':'review'},absent)
        try:
            tools.execute('grade',{'question_id':'gd-01','answer':'1.6'},absent)
        except ValueError:
            checks['missing_user_answer_not_invented'] = True
        else:
            raise AssertionError('Invented answer accepted')
        assert '错误' in present(state)['answer'] and not state['answer']
        checks['verified_results_render_without_model_prose'] = True
        try:
            tools.execute('record',{'question_id':'gd-01'},state)
        except ValueError:
            assert store.recent('demo') == []
        else:
            raise AssertionError('Unauthorized write')
        checks['unauthorized_write_rejected'] = True
        # Explain must cite evidence actually returned in this task.
        tools.execute('load_skill',{'name':'explain'},state)
        tools.execute('search',{'query':'梯度下降'},state)
        for ids in [[],['invented-citation']]:
            try:
                Agent.finish(action('finish',answer='解释',citations=ids),state)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid citation accepted')
        checks['citation_membership_and_presence'] = True
        sequence = [action('load_skill',{'name':'review'}),
                    action('grade',{'question_id':'gd-01','answer':'2'}),
                    action('record',{'question_id':'gd-01'}),
                    action('record',{'question_id':'gd-01'}),
                    action('finish',answer='答案不正确，已保存反馈。')]
        agent = Agent(ScriptedModel(sequence),tools,store)
        state = agent.create('gd-01 的答案是 2，保存',allow_record=True)
        state = agent.run(state,stop_after=3)
        assert state['status'] == 'running' and len(store.recent('demo')) == 1
        resumed = agent.run(store.load(state['task'],'demo'))
        assert resumed['status'] == 'done' and len(store.recent('demo')) == 1
        assert agent.run(store.load(state['task'],'demo'))['status'] == 'done'
        assert store.recent('other') == []
        try:
            store.load(state['task'],'other')
        except ValueError:
            pass
        else:
            raise AssertionError('Cross-learner resume')
        checks['resume_no_duplicate_and_learner_isolation'] = True
        workflow = Agent(ScriptedModel([action('load_skill',{'name':'review'}),
                         action('finish',answer='无记录'), action('progress'),
                         action('finish',answer='已查询记录')]),tools,store)
        workflow_state = workflow.run(workflow.create('查询记录',required_tools=['progress']))
        assert workflow_state['status'] == 'done' and 'error' in workflow_state['events'][1]
        assert present(workflow_state)['progress'][0]['question_id'] == 'gd-01'
        assert '已查询记录' not in present(workflow_state)['answer']
        checks['workflow_cannot_skip_required_tool'] = True
        # Force checkpoint failure after record INSERT: both must roll back.
        before = len(store.recent('demo'))
        rollback_state = agent.create('gd-01 的答案是 2，保存',allow_record=True)
        tools.execute('load_skill',{'name':'review'},rollback_state)
        tools.execute('grade',{'question_id':'gd-01','answer':'2'},rollback_state)
        try:
            with store.db:
                store.record(rollback_state,rollback_state['grades']['gd-01'])
                raise RuntimeError('simulated checkpoint failure')
        except RuntimeError:
            pass
        assert len(store.recent('demo')) == before
        checks['write_and_checkpoint_transaction_rollback'] = True
        looping = Agent(ScriptedModel([action('load_skill',{'name':'review'})]*8),tools,store)
        assert looping.run(looping.create('loop'))['status'] == 'exhausted'
        checks['bounded_loop'] = True
        invalid = Agent(ScriptedModel([{'action':'bad'}]),tools,store)
        assert invalid.run(invalid.create('bad JSON'))['status'] == 'failed'
        checks['invalid_model_action_is_failure'] = True
        # A malicious paragraph is merely search data, never an executable tool name.
        assert 'shell' not in tools.catalog['explain']['tools']
        checks['no_shell_capability'] = True
    save('verification.json',{'scope':'Offline engine tests with scripted model fixtures, not LLM success rate',
                              'source_files':len(index['sources']),'chunks':len(index['chunks']),'checks':checks})
    print(json.dumps(checks,indent=2))


if __name__ == '__main__':
    main()
