"""Bounded observe/act loop with committed checkpoints and explicit failures."""
import copy
import uuid
from model import validate_action


class Agent:
    def __init__(self, model, tools, store):
        self.model, self.tools, self.store = model, tools, store

    def create(self, request, learner='demo', allow_record=False, required_tools=()):
        learner = self.store.learner_id(learner)
        if not isinstance(request,str) or not 1 <= len(request) <= 1200:
            raise ValueError('问题长度需为 1～1200 字符')
        valid_tools = {'search','quiz','grade','record','progress'}
        if any(t not in valid_tools for t in required_tools):
            raise ValueError('未知的必需工具')
        if 'record' in required_tools and not allow_record:
            raise ValueError('要求 record 时必须授予保存权限')
        state = {'task':uuid.uuid4().hex,'learner':learner,'request':request,
                 'allow_record':bool(allow_record),'skill':None,'step':0,'status':'running',
                 'evidence':{},'quizzes':{},'grades':{},'recorded':[],'events':[],
                 'answer':'','citations':[], 'required_tools':list(dict.fromkeys(required_tools))}
        with self.store.db:
            self.store.checkpoint(state)
        return state

    @staticmethod
    def finish(action, state):
        if action['args'] or not action['answer'].strip():
            raise ValueError('finish 需要非空答案和空 args')
        if any(cid not in state['evidence'] for cid in action['citations']):
            raise ValueError('引用不属于本任务的检索结果')
        if state['skill'] == 'explain' and state['evidence'] and not action['citations']:
            raise ValueError('解释有检索结果时必须给出处')
        if state['skill'] == 'explain' and not state['evidence'] and '资料不足' not in action['answer']:
            raise ValueError('尚无资料，需要先搜索或说明资料不足')
        if not state['skill']:
            raise ValueError('请先加载技能')
        completed = {e['action']['action'] for e in state['events'] if 'observation' in e}
        pending = set(state.get('required_tools',[]))-completed
        if pending:
            raise ValueError('工作流必需工具尚未完成：'+','.join(sorted(pending)))
        if state['allow_record'] and state['grades'] and not state['recorded']:
            raise ValueError('本任务要求保存评分，请先调用 record')
        state.update(status='done', answer=action['answer'], citations=action['citations'])

    def run(self, state, stop_after=None):
        used = 0
        while state['status'] == 'running' and state['step'] < 8:
            candidate = copy.deepcopy(state)
            candidate['step'] += 1
            # Model I/O happens before the SQLite transaction, so no long lock is held.
            try:
                action = validate_action(self.model.decide(state))
            except Exception as exc:
                candidate['events'].append({'step':candidate['step'],'error':str(exc),'phase':'model'})
                candidate['status'] = 'failed'
                with self.store.db:
                    self.store.checkpoint(candidate)
                state = candidate
                break
            try:
                with self.store.db:
                    if action['action'] == 'finish':
                        self.finish(action, candidate)
                        observation = {'finished':True}
                    else:
                        observation = self.tools.execute(action['action'], action['args'], candidate)
                    candidate['events'].append({'step':candidate['step'],'action':action,'observation':observation})
                    self.store.checkpoint(candidate)
            except (ValueError, TypeError, KeyError) as exc:
                # Discard both partial in-memory state and database side effects.
                candidate = copy.deepcopy(state)
                candidate['step'] += 1
                candidate['events'].append({'step':candidate['step'],'action':action,'error':str(exc),'phase':'tool'})
                with self.store.db:
                    self.store.checkpoint(candidate)
            state = candidate
            used += 1
            if stop_after is not None and used >= stop_after:
                break
        if state['status'] == 'running' and state['step'] >= 8:
            state['status'] = 'exhausted'
            with self.store.db:
                self.store.checkpoint(state)
        return state
