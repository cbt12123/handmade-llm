"""Paragraph indexing and a transparent character-bigram lexical baseline."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import re
from common import ROOT, OUT

INDEX = OUT / 'course_index.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def terms(text):
    # Chinese bigrams do not require a segmentation library. English uses words.
    result = re.findall(r'[a-z0-9_]+', text.lower())
    for run in re.findall(r'[\u4e00-\u9fff]+', text):
        result.extend(run[i:i+2] for i in range(len(run)-1))
        if len(run) == 1:
            result.append(run)
    return result


def build_index():
    chunks, sources = [], {}
    for folder in sorted(ROOT.glob('第*部分*')):
        if not folder.name.startswith(('第一', '第二', '第三', '第四')):
            continue
        for path in sorted(folder.glob('*.md')):
            if path.name == 'README.md':
                continue
            relative = path.relative_to(ROOT).as_posix()
            sources[relative] = digest(path)
            heading, paragraph, first, fenced = '', [], 1, False

            def flush():
                if not paragraph:
                    return
                text = '\n'.join(paragraph).strip()
                # Keep every fragment. Long paragraphs are cut at a bounded size.
                for offset in range(0, len(text), 420):
                    fragment = text[offset:offset+420]
                    chunk_id = hashlib.sha256((relative+'\n'+heading+'\n'+fragment).encode()).hexdigest()[:12]
                    if not any(c['id'] == chunk_id for c in chunks):
                        chunks.append({'id':chunk_id, 'source':relative, 'heading':heading,
                                       'line':first, 'text':fragment})
                paragraph.clear()

            for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
                if line.startswith('```'):
                    flush(); fenced = not fenced; continue
                if fenced:
                    continue
                if line.startswith('#'):
                    flush(); heading = line.lstrip('#').strip(); continue
                if not line.strip() or line.startswith('!['):
                    flush(); continue
                if not paragraph:
                    first = line_number
                paragraph.append(line)
            flush()
    OUT.mkdir(parents=True, exist_ok=True)
    data = {'version':1, 'chunk_characters':420, 'sources':sources, 'chunks':chunks}
    INDEX.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    return data


class Knowledge:
    def __init__(self, path=INDEX):
        self.data = json.loads(Path(path).read_text(encoding='utf-8'))
        for name, expected in self.data['sources'].items():
            target = (ROOT / name).resolve()
            if not target.is_relative_to(ROOT) or not target.is_file() or digest(target) != expected:
                raise ValueError('知识源已变化，请重建索引：'+name)
        self.chunks = self.data['chunks']
        self.counts = [Counter(terms(c['heading']+' '+c['text'])) for c in self.chunks]
        self.df = Counter(t for counts in self.counts for t in counts)

    def search(self, query, limit=2):
        tokens = set(terms(query))
        if not tokens:
            return []
        ranked = []
        for chunk, counts in zip(self.chunks, self.counts):
            # A weak accidental overlap is not enough to return a source.
            if sum(t in counts for t in tokens) / len(tokens) < 0.5:
                continue
            # This is lexical relevance, not a learned semantic embedding.
            score = sum((1+math.log(counts[t])) * math.log(1+len(self.chunks)/(1+self.df[t]))
                        for t in tokens if counts[t]) / math.sqrt(max(1, sum(counts.values())))
            if query.strip().lower() in (chunk['heading']+' '+chunk['text']).lower():
                score += 2
            if score > 0:
                ranked.append({**chunk, 'score':round(score, 6)})
        return sorted(ranked, key=lambda c:(-c['score'], c['id']))[:limit]
