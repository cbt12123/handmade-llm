"""Source-grounded retrieval regressions and known real content failures."""
from pathlib import Path
import copy
import json
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'script'/'part05'))
from common import save
from knowledge import Knowledge, build_index, source_paths, sections
from content_review import review_explanation
from model import HTTPModel, action_schema
from toolbox import Toolbox


def rejected(path, root):
    try:
        Knowledge(path, root=root)
    except ValueError:
        return True
    raise AssertionError('Stale knowledge was accepted')


def main():
    index = build_index()
    knowledge = Knowledge()
    toolbox = Toolbox(knowledge, None)
    state = {'request':'解释梯度下降', 'skill':'explain', 'events':[],
             'evidence':{}, 'quizzes':{}, 'grades':{}, 'recorded':[],
             'allow_record':False, 'step':2}
    try:
        toolbox.execute('search', {'query':'梯'*25}, state)
    except ValueError:
        pass
    else:
        raise AssertionError('Oversized query accepted')
    observation = toolbox.execute('search', {'query':'梯度下降'}, state)
    state['events'] = [{'action':{'action':'search'}, 'observation':observation}]
    schema, _ = action_schema(state, toolbox.catalog)
    search = next(v for v in schema['oneOf'] if v['properties']['action']['const'] == 'search')
    assert search['properties']['args']['properties']['query']['maxLength'] == 24
    # Inspect the actual outgoing prompt without making a network call.
    class CapturedSession:
        def post(self, url, json, timeout):
            self.payload = json
            class Response:
                def raise_for_status(self):
                    pass
                def json(self):
                    return {'choices':[{'message':{'content':'{"action":"finish","args":{},"answer":"测试","citations":[]}'},'finish_reason':'stop'}]}
            return Response()
    model = HTTPModel()
    model.session.close()
    model.session = CapturedSession()
    model.decide(state)
    prompt = model.session.payload['messages'][1]['content']
    for chunk in observation['matches']:
        encoded_text = json.dumps(chunk['text'],ensure_ascii=False)[1:-1]
        assert prompt.count(encoded_text) == 1
    assert 'match_ids' in prompt and 'segments' not in prompt
    cases = json.loads((ROOT/'data/part05/retrieval_cases.json').read_text(encoding='utf-8'))
    results = []
    for case in cases:
        hits = knowledge.search(case['query'])
        if case.get('empty'):
            passed = not hits
        else:
            relevant = [h for h in hits if any(p in h['source'] for p in case['sources'])]
            passed = bool(relevant) and all(any(term in h['text'] for h in relevant) for term in case['contains'])
        results.append({'query':case['query'], 'passed':passed, 'hits':hits})
    # Compare complete fenced formulas, not just a count of opening markers.
    expected_math, retained_math = set(), set()
    for path in source_paths():
        relative = path.relative_to(ROOT).as_posix()
        raw = path.read_text(encoding='utf-8')
        expected_math.update((relative, m.group(0)) for m in re.finditer(
            r'^```math[ \t]*\n.*?^```[ \t]*(?=\n|$)', raw, flags=re.M|re.S))
    for chunk in index['chunks']:
        lines = (ROOT/chunk['source']).read_text(encoding='utf-8').splitlines()
        segments = ['\n'.join(lines[s['line']-1:s['end_line']]) for s in chunk['segments']]
        assert '\n\n'.join(segments) == chunk['text']
        retained_math.update((chunk['source'], text) for text in segments if text.lstrip().startswith('```math'))
    assert expected_math == retained_math
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        chapter = root/'第一部分-测试'/'01.md'
        chapter.parent.mkdir()
        chapter.write_text('# 测试\n\n## 定义\n\n公式前解释。\n\n```math\ny=x^2\n```\n\n公式后符号说明。\n\n```python\nSECRET_EXECUTABLE\n```\n\n```text\n输入 → 输出\n```\n', encoding='utf-8')
        path = root/'index.json'
        fixture = build_index(root, path)
        chunk = fixture['chunks'][0]
        assert all(t in chunk['text'] for t in ['公式前解释', 'y=x^2', '公式后符号说明', '输入 → 输出'])
        assert 'SECRET_EXECUTABLE' not in chunk['text']
        Knowledge(path, root)
        extra = chapter.parent/'02.md'
        extra.write_text('# 新章节\n\n新内容', encoding='utf-8')
        assert rejected(path, root)
        extra.unlink()  # Only the explicitly created temporary fixture.
        chapter.write_text(chapter.read_text(encoding='utf-8')+'\n新内容', encoding='utf-8')
        assert rejected(path, root)
        build_index(root, path)
        chapter.unlink()
        assert rejected(path, root)
        try:
            list(sections('# Bad\n```math\nx'))
        except ValueError:
            pass
        else:
            raise AssertionError('Unclosed formula accepted')
    old = json.loads((ROOT/'data/part05/content_regression.json').read_text(encoding='utf-8'))
    case = {'expected_terms':['梯度','学习率']}
    old_screen = review_explanation(case,old['state'])
    assert old['legacy_passed'] and not old_screen['screening_passed'] and old_screen['flags']
    positive = copy.deepcopy(old['state'])
    positive['answer'] = '正学习率下，更新方向为负梯度，学习率控制步长。梯度下降不能保证找到全局最小值。'
    assert review_explanation(case,positive)['screening_passed']
    wrong = copy.deepcopy(positive)
    wrong['answer'] = '梯度下降一定能找到全局最小值，学习率控制步长。'
    assert not review_explanation(case,wrong)['screening_passed']
    assert not review_explanation({**case,'expected_sources':['30-归约']},positive)['screening_passed']
    extra = copy.deepcopy(positive)
    extra['answer'] = old['additional_observed_answer']
    assert not review_explanation(case,extra)['screening_passed']
    checks = {'complete_math_and_exact_source_ranges':True,
              'query_limit_in_schema_and_host':True,
              'retrieval_text_appears_once_in_model_prompt':True,
              'formula_neighbors_and_nonexecutable_text':True,
              'new_modified_removed_sources_rejected':True,
              'unclosed_formula_rejected':True,
              'known_real_wrong_answer_rejected':True,
              'observed_indirect_direction_claim_rejected':True,
              'correct_negated_guarantee_not_flagged':True,
              'wrong_expected_source_rejected':True}
    report = {'scope':'Authored lexical cases including observed model queries, and deterministic regressions; not general retrieval accuracy or model correctness',
              'index_version':index['version'], 'source_files':len(index['sources']), 'chunks':len(index['chunks']),
              'complete_math_blocks':len(expected_math), 'passed':sum(r['passed'] for r in results),
              'count':len(results), 'queries':results, 'checks':checks,
              'known_real_answer_screen':old_screen}
    save('24_retrieval_after.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('queries', 'known_real_answer_screen')},ensure_ascii=False,indent=2))
    assert all(r['passed'] for r in results)


if __name__ == '__main__':
    main()
