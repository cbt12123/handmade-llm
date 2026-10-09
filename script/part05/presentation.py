"""Render verified tool facts independently of model prose."""


def present(state):
    parts = []
    if state['quizzes']:
        parts.append('练习（来自题库）：\n'+'\n'.join(f'{qid}：{text}' for qid,text in state['quizzes'].items()))
    if state['grades']:
        parts.append('确定评分：\n'+'\n'.join(f"{qid}：{'正确' if grade['correct'] else '错误'}。{grade['feedback']}"
                                           for qid,grade in state['grades'].items()))
        parts.append('记录状态：'+('已保存 '+','.join(state['recorded']) if state['recorded'] else '未保存'))
    progress = [e['observation']['records'] for e in state['events'] if e.get('action',{}).get('action') == 'progress' and 'observation' in e]
    if progress:
        records = progress[-1]
        parts.insert(0,'最近学习记录：\n'+('\n'.join(f"{r['question_id']}：{'正确' if r['correct'] else '错误'}。{r['feedback']}" for r in records) or '尚无记录'))
    else:
        records = None
    # Structured tasks render authoritative tool facts. Unchecked prose must
    # not add invented grades or records; it stays in state/logs for diagnosis.
    if state['answer'] and not (state['quizzes'] or state['grades'] or progress):
        parts.append(state['answer'])
    return {'task':state['task'],'status':state['status'],'answer':'\n\n'.join(parts),
            'questions':[{'id':k,'question':v} for k,v in state['quizzes'].items()],
            'grades':list(state['grades'].values()),'recorded':state['recorded'],
            'progress':records,
            'sources':[state['evidence'][cid] for cid in state['citations']],
            'errors':[e['error'] for e in state['events'] if 'error' in e]}
