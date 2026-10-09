"""Normalize authored Markdown math, fence spacing and heading-only separation."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
docs = [ROOT / 'README.md', *ROOT.glob('第*部分*/*.md')]
changed = []
for path in docs:
    old = path.read_text(encoding='utf-8')
    lines = old.splitlines()
    result = []
    fenced = False
    display = None
    for line in lines:
        if display is not None:
            if '$$' in line:
                display.append(line.split('$$', 1)[0])
                result += ['```math', *display, '```', '']
                display = None
            else:
                display.append(line)
            continue
        if line.lstrip().startswith('```'):
            fenced = not fenced
            result.append(line)
            continue
        if fenced:
            result.append(line)
            continue
        if re.fullmatch(r'\s*(?:-{3,}|\*{3,}|_{3,})\s*', line):
            continue
        if '$$' in line:
            before, tail = line.split('$$', 1)
            if before.strip(): result.append(before.rstrip())
            if '$$' in tail:
                expression, after = tail.split('$$', 1)
                result += ['', '```math', expression.strip(), '```', '']
                if after.strip(): result.append(after.strip())
            else:
                result.append('')
                display = [tail.strip()] if tail.strip() else []
            continue
        line = re.sub(r'(?<!\\)\$([^$\n]+)\$',
                      lambda m: m.group(0) if m.group(1).startswith('`') else '$`'+m.group(1)+'`$', line)
        result.append(line)
    if display is not None or fenced:
        raise ValueError(f'Unclosed math or code fence: {path}')
    new = re.sub(r'\n{3,}', '\n\n', '\n'.join(result)).rstrip() + '\n'
    if new != old:
        path.write_text(new, encoding='utf-8')
        changed.append(str(path.relative_to(ROOT)))
print('Normalized:', len(changed), 'documents')
