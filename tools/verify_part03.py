"""Run actual chapter outputs, then validate local links and reproducible checks."""
from pathlib import Path
import subprocess,sys,json,re,shutil,tempfile,os,contextlib,io
ROOT=Path(__file__).resolve().parents[1]; PART=ROOT/'第三部分-机器学习与深度学习'; OUT=ROOT/'outputs'/'part03'
scripts=['08_sklearn_task.py','09_numpy_network.py','10_torch_training.py','11_numpy_cnn.py','11_cnn.py','12_text.py','13_transformer.py']
if '--extras' in sys.argv: scripts+=['12_real_sms.py','11_transfer_learning.py']
for name in ([] if '--check-only' in sys.argv else scripts):
    print('Running',name,flush=True); subprocess.run([sys.executable,str(ROOT/'script'/'part03'/name)],cwd=ROOT,check=True)
if '--check-only' not in sys.argv:
    subprocess.run([sys.executable,str(ROOT/'script'/'part03'/'11_cnn.py'),'--image',str(OUT/'11_example_digit.png')],check=True)
    subprocess.run([sys.executable,str(ROOT/'script'/'part03'/'learning_card_demo.py'),'--image',str(OUT/'11_example_digit.png'),'--feedback','这道题不懂需要复习','--digits','2','4','7','9','1'],check=True)
for name in ['08-predictions.png','09-training.png','09-boundary.png','09-digits-training.png','10_curves.png','11_numpy_training.png','11_feature_maps.png','11_curves.png','13_training_attention.png']:
    path=OUT/name
    if path.exists(): shutil.copy2(path,ROOT/'images'/('part03-result-'+name))
checked=0
examples=0
for doc in [ROOT/'README.md',*PART.glob('*.md')]:
    text=doc.read_text(encoding='utf-8')
    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
        if '://' in target or target.startswith('#'): continue
        assert (doc.parent/target.split('#')[0]).exists(), (doc.name,target)
        checked+=1
    assert text.count('```')%2==0,doc
    # Chapter snippets share variables within a chapter, matching reading order.
    namespace={'__name__':'__main__'}; previous=Path.cwd()
    with tempfile.TemporaryDirectory(prefix='part03-examples-') as temp:
        try:
            os.chdir(temp)
            for snippet in re.findall(r'```python\s*\n(.*?)\n```',text,re.S):
                with contextlib.redirect_stdout(io.StringIO()): exec(compile(snippet,str(doc),'exec'),namespace)
                examples+=1
        finally: os.chdir(previous)
reports={p.name:json.loads(p.read_text(encoding='utf-8')) for p in OUT.glob('*.json') if p.name!='verification.json'}
for name,data in reports.items():
    if 'reload_equal' in data: assert data['reload_equal'],name
    if 'reload_identical' in data: assert data['reload_identical'],name
assert reports['09-report.json']['gradient_max_error']<1e-6
assert max(reports['11_numpy_report.json']['gradient_checks']['sampled_finite_difference_max_errors'].values())<1e-6
assert reports['11_numpy_report.json']['gradient_checks']['all_derivatives_torch_reference_max_error']<1e-10
assert reports['10_report.json']['numpy_autograd_max_error']<1e-10
assert reports['10_report.json']['resume_one_step_loss']>0
assert reports['09-report.json']['digits_binary']['reload_equal']
assert reports['13_report.json']['causal_prefix_check']
assert reports['learning_card_demo.json']['reversal_exact_including_eos']
assert reports['learning_card_demo.json']['feedback_predicted_label'] in [0,1]
prior=json.loads((OUT/'verification.json').read_text(encoding='utf-8')) if (OUT/'verification.json').exists() else {}
summary={'python':sys.version,'scripts_run':prior.get('scripts_run',[]) if '--check-only' in sys.argv else scripts+['learning_card_demo.py'],'local_links_checked':checked,'python_examples_executed':examples,'reports':sorted(reports),'checks':['finite-difference gradient','NumPy/PyTorch gradient parity','checkpoint reload','resume one step','CNN image prediction','3/8 NumPy classifier','causal prefix invariance','masked attention','free generation','learning-card integrated CLI']}
(OUT/'verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8'); print('Verified',summary,flush=True)
