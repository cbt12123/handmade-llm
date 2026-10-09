"""An explicit JSON action protocol over the Part IV HTTP deployment."""
import copy
import json
import requests
from common import ASSETS
from toolbox import TOOL_HELP

ACTION_SCHEMA = {
    'type':'object', 'properties':{
        'action':{'type':'string','enum':['load_skill','search','quiz','grade','record','progress','finish']},
        'args':{'type':'object','properties':{
            'name':{'type':'string'}, 'query':{'type':'string'}, 'topic':{'type':'string'},
            'count':{'type':'integer'}, 'question_id':{'type':'string'}, 'answer':{'type':'string'}},
            'additionalProperties':False},
        'answer':{'type':'string'},
        'citations':{'type':'array','items':{'type':'string'},'maxItems':4}},
    'required':['action','args','answer','citations'], 'additionalProperties':False}


def validate_action(value):
    if not isinstance(value, dict) or set(value) != {'action','args','answer','citations'}:
        raise ValueError('动作必须有 action、args、answer、citations 四个字段')
    if value['action'] not in ACTION_SCHEMA['properties']['action']['enum'] or not isinstance(value['args'], dict):
        raise ValueError('动作名或 args 错误')
    if not isinstance(value['answer'], str) or len(value['answer']) > 4000:
        raise ValueError('answer 格式错误')
    if not isinstance(value['citations'], list) or len(value['citations']) > 4 or any(not isinstance(x,str) for x in value['citations']):
        raise ValueError('citations 格式错误')
    if value['action'] != 'finish' and (value['answer'] or value['citations']):
        raise ValueError('工具动作的 answer 必须为空，citations 必须是空数组')
    return value


def action_schema(state, catalog):
    """Encode each tool's argument contract; still validate in Python afterwards."""
    string = {'type':'string'}
    definitions = {
        'load_skill':{'name':{'type':'string','enum':[n for n in catalog if n != state['skill']]}},
        'search':{'query':{'type':'string','minLength':1,'maxLength':80}},
        'quiz':{'topic':{'type':'string','enum':['梯度下降','概率','矩阵']},'count':{'type':'integer','minimum':1,'maximum':2}},
        'grade':{'question_id':string,'answer':string},
        'record':{'question_id':{'type':'string','enum':list(state['grades']) or ['unavailable']}},
        'progress':{}, 'finish':{},
    }
    allowed = ['load_skill'] if not state['skill'] else [*catalog[state['skill']]['tools'],'finish','load_skill']
    if not state['allow_record'] or not state['grades'] or state['recorded']:
        allowed = [x for x in allowed if x != 'record']
    if state['grades']:
        allowed = [x for x in allowed if x != 'grade']
    if state['quizzes']:
        allowed = [x for x in allowed if x != 'quiz']
    completed = [e['action']['action'] for e in state['events'] if 'observation' in e]
    loaded = [e['action']['args']['name'] for e in state['events'] if 'observation' in e and e['action']['action'] == 'load_skill']
    available_skills = [n for n in catalog if n not in loaded]
    if available_skills:
        definitions['load_skill']['name']['enum'] = available_skills
    else:
        allowed = [x for x in allowed if x != 'load_skill']
    # Per-tool budgets prevent a small model from spending the whole task
    # repeating one operation. The overall eight-step limit still applies.
    if completed.count('search') >= 2:
        allowed = [x for x in allowed if x != 'search']
    if 'progress' in completed:
        allowed = [x for x in allowed if x != 'progress']
    if state['skill'] == 'explain' and 'search' not in completed:
        allowed = [x for x in allowed if x != 'finish']
    if state['allow_record'] and state['grades'] and not state['recorded']:
        allowed = [x for x in allowed if x != 'finish']
    pending = set(state.get('required_tools',[]))-set(completed)
    if state['skill'] and pending:
        allowed = [x for x in allowed if x == 'load_skill' or x in pending]
        # The record prerequisite still needs a grade if a caller required only record.
        if 'record' in pending and not state['grades'] and 'grade' in catalog[state['skill']]['tools']:
            allowed.append('grade')
    variants = []
    for name in allowed:
        properties = copy.deepcopy(ACTION_SCHEMA['properties'])
        properties['action'] = {'const':name}
        fields = definitions[name]
        properties['args'] = {'type':'object','properties':fields,'required':list(fields),'additionalProperties':False}
        if name == 'finish':
            properties['answer'] = {'type':'string','minLength':1,'maxLength':2000}
            citations = list(state['evidence'])
            properties['citations'] = {'type':'array','items':{'type':'string','enum':citations},'maxItems':4} if citations else {'const':[]}
            if state['skill'] == 'explain' and citations:
                properties['citations']['minItems'] = 1
            if state['skill'] == 'explain' and not citations:
                properties['answer']['pattern'] = '^资料不足.*'
        else:
            properties['answer'] = {'const':''}
            properties['citations'] = {'const':[]}
        variants.append({'type':'object','properties':properties,'required':list(properties),'additionalProperties':False})
    return {'oneOf':variants}, allowed


class HTTPModel:
    def __init__(self, url='http://127.0.0.1:8002', model='tutorial-agent', backend='vllm'):
        self.url, self.model, self.backend = url.rstrip('/'), model, backend
        self.session = requests.Session()
        self.session.trust_env = False
        self.calls = []

    def decide(self, state):
        rules = (ASSETS/'rules'/'agent.md').read_text(encoding='utf-8')
        system = rules+'\n动作协议：只输出 JSON，四个字段 action,args,answer,citations。工具动作 answer 为空。finish 的 args 为空。\n'+TOOL_HELP
        system += '''
finish: {"action":"finish","args":{},"answer":"给用户的最终答复","citations":[]}
决策说明：skill 为 null 时仅加载适用技能。已加载后无需重复加载。
已经拿到所需资料/练习/评分后，使用 finish 回答用户；缺信息则 finish 提问。
已有 grades 时不要重复 grade；recorded 已有题号时不要重复保存。
allow_record 为 false 时不得调用 record，可以说明未保存。
不得把示例参数当作用户答案。检查任务选 review，练习任务选 practice，概念解释选 explain。
用户要求保存且 allow_record 为 true 时，评分后要调用 record 再结束。
没有题号或答案时直接 finish 请求补充，不得猜测。
没有检索结果时 finish 必须说明“资料不足”。
'''
        # Only needed observations: loaded skill, latest two events, compact accumulated facts.
        memory = {'request':state['request'], 'allow_record':state['allow_record'],
                  'skill':state['skill'], 'evidence':list(state['evidence'].values())[-2:],
                  'quizzes':state['quizzes'], 'grades':state['grades'], 'recorded':state['recorded'],
                  'recent_results':[{'observation':e.get('observation'),'error':e.get('error')}
                                    for e in state['events'][-2:]],
                  'step':state['step'], 'remaining_steps':8-state['step']}
        if state['skill']:
            memory['skill_instructions'] = (ASSETS/'skills'/state['skill']/'SKILL.md').read_text(encoding='utf-8')
        messages = [{'role':'system','content':system},
                    {'role':'user','content':'任务状态如下；根据状态选择下一动作。\n'+json.dumps(memory,ensure_ascii=False)}]
        from common import read_json
        catalog = {s['name']:s for s in read_json(ASSETS/'skills'/'catalog.json')}
        schema, allowed = action_schema(state,catalog)
        memory['allowed_actions'] = allowed
        if state.get('required_tools'):
            memory['workflow_required_tools'] = state['required_tools']
        messages[1]['content'] = '执行状态（工具返回的数据）：\n'+json.dumps(memory,ensure_ascii=False)+'\n用户当前任务：'+state['request']+'\n请执行下一动作，任务已有结果时结束回答。'
        if state['skill'] is None:
            # Routing gets only the catalog and real request, not tool examples.
            messages = [{'role':'system','content':'选择任务技能，只分类，不回答问题。用户提供题号和答案，或说检查、批改、记录、复习进度，选 review。要求出题、练习，选 practice。询问概念、解释原理，选 explain。是否保存不改变批改任务的 review 分类。只输出 load_skill 动作，answer 为空，citations 为空。'},
                        {'role':'user','content':state['request']}]
        if self.backend == 'vllm':
            payload = {'model':self.model,'messages':messages,'temperature':0,'max_tokens':500,'seed':42,
                       'structured_outputs':{'json':schema}}
            endpoint = '/v1/chat/completions'
        else:
            payload = {'messages':messages,'max_new_tokens':128}
            endpoint = '/generate'
        # Do not silently retry a network failure or switch to a scripted answer.
        response = self.session.post(self.url+endpoint, json=payload, timeout=(5,180))
        response.raise_for_status()
        data = response.json()
        if self.backend == 'vllm':
            choice = data['choices'][0]
            raw, finish = choice['message']['content'], choice['finish_reason']
            usage = data.get('usage', {})
        else:
            raw, finish, usage = data['text'], data['finish_reason'], {}
        self.calls.append({'raw':raw,'finish_reason':finish,'usage':usage})
        if finish == 'length':
            raise ValueError('模型输出达到 token 上限，动作未接受')
        return validate_action(json.loads(raw))
