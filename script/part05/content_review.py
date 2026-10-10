"""Narrow factual screening; never a general semantic correctness judge."""
import re


def review_explanation(case, state):
    answer = state['answer']
    citations = state['citations']
    cited = [state['evidence'][cid] for cid in citations if cid in state['evidence']]
    terms_ok = all(term.lower() in answer.lower() for term in case.get('expected_terms', []))
    sources_ok = bool(citations) and len(cited) == len(citations)
    expected_sources = case.get('expected_sources', [])
    relevant_source = not expected_sources or any(
        any(pattern in chunk['source'] for pattern in expected_sources) for chunk in cited)
    required_evidence = case.get('required_evidence', [])
    evidence_ok = all(any(term in chunk['text'] for chunk in cited) for term in required_evidence)
    flags = []
    # This explicitly covers the previously observed error; it is not entailment.
    if re.search(r'学习率[^。！？\n]{0,20}(?:决定|改变|控制)[^。！？\n]{0,45}(?:方向|向前|向后)', answer):
        flags.append('learning_rate_direction_claim_needs_review')
    for sentence in re.split(r'[。！？\n]', answer):
        if re.search(r'(?:保证|一定|必然)[^。]{0,25}全局(?:最小|极小)', sentence):
            if not re.search(r'不|不能|无法|未必|并非', sentence):
                flags.append('global_minimum_guarantee_needs_review')
    checks = {'task_done':state['status'] == 'done', 'citation_membership':sources_ok,
              'expected_terms':terms_ok, 'expected_source':relevant_source,
              'required_evidence_present':evidence_ok, 'no_known_flags':not flags}
    return {'screening_passed':all(checks.values()), 'checks':checks, 'flags':flags,
            'human_review_required':True,
            'scope':'Term/source checks and two narrow contradiction patterns; quotation, negation and unseen errors require human review.'}
