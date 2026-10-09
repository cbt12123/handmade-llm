"""Validate authored Markdown and build local review pages (no GitHub writes).

Optional preview dependencies:
python -m pip install markdown-it-py==3.0.0 mdit-py-plugins==0.4.2
MathJax 3.2.2 is downloaded once for local math rendering.
"""
from pathlib import Path
import json
import re
import html
import unicodedata
from urllib.parse import unquote
from urllib.request import build_opener, ProxyHandler, Request
from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'review'
OUT.mkdir(parents=True, exist_ok=True)
DOCS = [ROOT / 'README.md', *sorted(ROOT.glob('第*部分*/*.md')), *sorted(ROOT.glob('assets/**/*.md'))]
parser = MarkdownIt('commonmark', {'html': False}).enable('table').use(dollarmath_plugin)

def slug(text):
    text = re.sub(r'[`*_]', '', text.lower())
    return ''.join(c for c in text if c in ' -' or unicodedata.category(c)[0] in 'LN').replace(' ', '-')

def outside_fences(text):
    return re.sub(r'```[^\n]*\n.*?^```[^\n]*$', '', text, flags=re.S|re.M)

assets = OUT / 'assets'
assets.mkdir(exist_ok=True)
mathjax = assets / 'tex-svg.js'
if not mathjax.exists():
    url = 'https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js'
    opener = build_opener(ProxyHandler({'http': 'http://127.0.0.1:7897', 'https': 'http://127.0.0.1:7897'}))
    try:
        with opener.open(Request(url, headers={'User-Agent':'tutorial-review'}), timeout=30) as response:
            mathjax.write_bytes(response.read())
    except Exception:
        # Allow environments without the author's local proxy to use a direct path.
        with build_opener(ProxyHandler({})).open(url, timeout=30) as response:
            mathjax.write_bytes(response.read())

pages = {path: OUT / ('index.html' if path == ROOT/'README.md' else path.stem+'.html') for path in DOCS}
# Part READMEs need distinct filenames.
for path in DOCS:
    if path.name == 'README.md' and path != ROOT/'README.md': pages[path] = OUT/(path.parent.name+'.html')
    if path.is_relative_to(ROOT/'assets'):
        pages[path] = OUT/('asset-'+'-'.join(path.relative_to(ROOT/'assets').with_suffix('').parts)+'.html')

summary = {'documents':len(DOCS), 'local_links':0, 'math_blocks':0, 'inline_math':0,
           'tables':0, 'horizontal_rules':0, 'errors':[], 'renderer':'CommonMark tables + MathJax 3.2.2; local approximation, not live GitHub'}

def issue(path, message):
    summary['errors'].append(f'{path.relative_to(ROOT)}: {message}')

for path in DOCS:
    text = path.read_text(encoding='utf-8')
    if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', text): issue(path,'unexpected control character')
    if '\ufffd' in text: issue(path,'Unicode replacement character')
    plain = outside_fences(text)
    if '$$' in plain: issue(path,'display math not normalized')
    if re.search(r'^\s*(?:-{3,}|\*{3,}|_{3,})\s*$',plain,re.M): issue(path,'horizontal rule')
    # GitHub's protected inline delimiter is not a native dollarmath convention.
    local_text = re.sub(r'\$`([^\n]*?)`\$', lambda m:'$'+m.group(1)+'$', text)
    tokens = parser.parse(local_text)
    if sum(t.type == 'heading_open' and t.tag == 'h1' for t in tokens) != 1: issue(path,'expected one h1')
    used = {}
    for i, token in enumerate(tokens):
        if token.type == 'heading_open':
            heading = slug(tokens[i+1].content)
            number = used.get(heading,0); used[heading] = number+1
            token.attrSet('id', heading if number==0 else f'{heading}-{number}')
        if token.type == 'hr': summary['horizontal_rules']+=1
        if token.type == 'table_open': summary['tables']+=1
        if token.type == 'fence' and token.info.strip() == 'math': summary['math_blocks']+=1
        if token.type == 'inline':
            summary['inline_math'] += sum(t.type == 'math_inline' for t in token.children or [])
        for child in [token, *(token.children or [])]:
            attribute = 'src' if child.type == 'image' else 'href' if child.type == 'link_open' else None
            if attribute:
                target = child.attrGet(attribute)
                if target.startswith(('https://','http://','mailto:')): continue
                url, _, anchor = target.partition('#')
                resolved = (path.parent/unquote(url)).resolve() if url else path
                if not resolved.is_file(): issue(path, f'missing target: {target}')
                elif anchor and resolved.suffix == '.md':
                    headings = re.findall(r'^#{1,6} (.+)$', outside_fences(resolved.read_text(encoding='utf-8')), re.M)
                    if unquote(anchor) not in {slug(h) for h in headings}: issue(path, f'missing heading: {target}')
                if resolved in pages: child.attrSet(attribute,pages[resolved].name+('#'+anchor if anchor else ''))
                else: child.attrSet(attribute, resolved.as_uri())
                summary['local_links']+=1
    body = parser.renderer.render(tokens, parser.options, {})
    # Math blocks are intentionally rendered by MathJax, not syntax highlighting.
    body = re.sub(r'<pre><code class="language-math">(.*?)</code></pre>',
                  lambda m:'<div class="math-block">\\['+m.group(1)+'\\]</div>',body,flags=re.S)
    body = re.sub(r'<span class="math inline">(.*?)</span>',
                  lambda m:'<span>\\('+m.group(1)+'\\)</span>',body,flags=re.S)
    template = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>本地审阅</title>
<style>body{margin:0;background:#f6f8fa;color:#1f2328;font-family:"Microsoft YaHei",sans-serif;line-height:1.85}main{max-width:940px;margin:28px auto;padding:32px 44px;background:white;border:1px solid #d1d9e0;border-radius:10px}h1,h2,h3{line-height:1.45;margin-top:1.4em}h1{font-size:30px}h2{font-size:24px}h3{font-size:19px}p{margin:16px 0}a{color:#0969da}img{max-width:100%;height:auto}pre{padding:16px;background:#f6f8fa;border-radius:6px;overflow-x:auto;line-height:1.55}code{font-family:Consolas,monospace;font-size:14px}p code,td code{background:#eff1f3;padding:2px 4px;border-radius:3px}table{border-collapse:collapse;display:block;overflow-x:auto;margin:20px 0}td,th{border:1px solid #d1d9e0;padding:8px 12px}th{background:#f6f8fa}li{margin:5px 0}.math-block{overflow-x:auto;margin:24px 0}.notice{font-size:14px;color:#59636e;background:#eef5fc;padding:12px;border-radius:6px}mjx-container{max-width:100%}@media(max-width:700px){main{padding:18px;margin:8px}h1{font-size:25px}h2{font-size:21px}}</style>
<script>window.MathJax={tex:{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']],processEscapes:true},svg:{fontCache:'local'}};</script><script defer src="assets/tex-svg.js"></script>
<main><p class="notice">本地审阅：CommonMark表格与MathJax公式。当前修订未上传；此页不等同于GitHub线上页面。<a href="index.html">课程目录</a></p>BODY</main></html>'''
    pages[path].write_text(template.replace('BODY',body),encoding='utf-8')

(OUT/'markdown-report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
if summary['errors']: raise SystemExit(1)
