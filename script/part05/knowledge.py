"""Section-aware lexical retrieval with complete math blocks and provenance."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import re
from common import ROOT, OUT

INDEX = OUT / 'course_index.json'
INDEX_VERSION = 2
CHUNK_TARGET = 700
OVERLAP_TARGET = 200


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


def source_paths(root=ROOT):
    chapters = [p for folder in sorted(root.glob('第*部分*')) if folder.is_dir()
                for p in sorted(folder.glob('*.md')) if p.name != 'README.md']
    return sorted([*chapters, *(root / '附录').glob('*.md')])


def sections(text):
    """Keep math/text fences atomic; omit executable code and image-only lines."""
    heading, units, paragraph = '', [], []
    paragraph_start = 1
    fence, language, math_lines, math_start = None, '', [], 1

    def flush(last):
        if paragraph:
            units.append({'text':'\n'.join(paragraph), 'line':paragraph_start, 'end_line':last})
            paragraph.clear()

    for number, line in enumerate(text.splitlines(), 1):
        marker = re.match(r'^\s{0,3}(`{3,}|~{3,})(.*)$', line)
        if fence:
            if language in ('math', 'text'):
                math_lines.append(line)
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                if language in ('math', 'text'):
                    units.append({'text':'\n'.join(math_lines), 'line':math_start, 'end_line':number})
                fence, language, math_lines = None, '', []
            continue
        if marker:
            flush(number-1)
            fence, language = marker[1], marker[2].strip().lower()
            if language in ('math', 'text'):
                math_lines, math_start = [line], number
            continue
        title = re.match(r'^#{1,6}\s+(.+)$', line)
        if title:
            flush(number-1)
            if units:
                yield heading, units
            heading, units = title[1].strip(), []
        elif not line.strip() or line.lstrip().startswith('!['):
            flush(number-1)
        else:
            if not paragraph:
                paragraph_start = number
            paragraph.append(line)
    if fence:
        raise ValueError('未闭合的 Markdown 围栏，不能建立知识索引')
    flush(len(text.splitlines()))
    if units:
        yield heading, units


def windows(units):
    """Adjacent paragraphs overlap; an individual formula is never sliced."""
    current = []
    for unit in units:
        size = sum(len(u['text'])+2 for u in current)
        if current and size+len(unit['text']) > CHUNK_TARGET:
            yield current
            overlap, length = [], 0
            for previous in reversed(current):
                if length+len(previous['text'])+2 > OVERLAP_TARGET:
                    break
                overlap.insert(0, previous)
                length += len(previous['text'])+2
            current = overlap
        current.append(unit)
    if current:
        yield current


def build_index(root=ROOT, output=INDEX):
    chunks, sources, ids = [], {}, set()
    for path in source_paths(root):
        relative = path.relative_to(root).as_posix()
        sources[relative] = digest(path)
        document_text = path.read_text(encoding='utf-8')
        document_heading = next((line[2:].strip() for line in document_text.splitlines() if line.startswith('# ')), '')
        for heading, units in sections(document_text):
            for group in windows(units):
                text = '\n\n'.join(u['text'] for u in group)
                chunk_id = hashlib.sha256((relative+'\n'+heading+'\n'+text).encode()).hexdigest()[:12]
                if chunk_id in ids:
                    continue
                ids.add(chunk_id)
                chunks.append({'id':chunk_id, 'source':relative, 'heading':heading, 'document_heading':document_heading,
                               'line':group[0]['line'], 'end_line':group[-1]['end_line'],
                               'segments':[{'line':u['line'], 'end_line':u['end_line']} for u in group],
                               'text':text})
    data = {'version':INDEX_VERSION, 'chunk_target_characters':CHUNK_TARGET,
            'overlap_target_characters':OVERLAP_TARGET,
            'scope':'All six parts and learning appendices; math/text retained, executable fences excluded',
            'sources':sources, 'chunks':chunks}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    return data


class Knowledge:
    def __init__(self, path=INDEX, root=ROOT):
        self.data = json.loads(Path(path).read_text(encoding='utf-8'))
        if self.data.get('version') != INDEX_VERSION:
            raise ValueError('知识索引格式已升级，请重建索引')
        actual = {p.relative_to(root).as_posix() for p in source_paths(root)}
        if actual != set(self.data['sources']):
            raise ValueError('知识文件集合已变化，请重建索引')
        for name, expected in self.data['sources'].items():
            target = (root / name).resolve()
            if not target.is_relative_to(root.resolve()) or not target.is_file() or digest(target) != expected:
                raise ValueError('知识源已变化，请重建索引：'+name)
        self.chunks = self.data['chunks']
        self.counts = [Counter(terms(c['heading']+' '+c['text'])) for c in self.chunks]
        self.df = Counter(t for counts in self.counts for t in counts)

    def search(self, query, limit=2):
        tokens = set(terms(query))
        tokens -= {'and', 'vs', 'the', 'of', 'is', 'are', 'what', 'do'}
        if not tokens or limit < 1:
            return []
        normalize = lambda s: re.sub(r'\s+', '', s.lower())
        needles = {normalize(query)}
        aliases = {'kv缓存':'KV Cache', 'kvcache':'KV缓存'}
        for spelling, alias in aliases.items():
            if spelling in normalize(query):
                needles.update({spelling, normalize(alias)})
                tokens = set(terms('KV缓存'))
        # A platform qualifier must not overwhelm the named operation.
        platforms = tokens & {'gpu', 'cpu'}
        if len(platforms) == 1 and tokens-platforms:
            tokens -= platforms
            for platform in platforms:
                if normalize(query).startswith(platform):
                    needles.add(normalize(query)[len(platform):])
        ranked = []
        for chunk, counts in zip(self.chunks, self.counts):
            # A weak accidental overlap is not enough to return a source.
            if sum(t in counts for t in tokens) / len(tokens) < 0.5:
                continue
            # This is lexical relevance, not a learned semantic embedding.
            score = sum((1+math.log(counts[t])) * math.log(1+len(self.chunks)/(1+self.df[t]))
                        for t in tokens if counts[t]) / math.sqrt(max(1, sum(counts.values())))
            if any(n in normalize(chunk['heading']+' '+chunk['text']) for n in needles):
                score += 2
            if any(n in normalize(chunk['heading']) for n in needles):
                score += 2
            if any(n in normalize(chunk.get('document_heading', '')) for n in needles):
                score += 0.75
            if chunk['heading'] == chunk.get('document_heading'):
                score *= 0.7
            if re.search(r'产出|随堂|自检|自测|入口', chunk['heading']):
                score *= 0.5
            if score > 0:
                ranked.append({**chunk, 'score':round(score, 6)})
        results, selected_sections = [], set()
        for chunk in sorted(ranked, key=lambda c:(-c['score'], c['id'])):
            key = (chunk['source'], chunk['heading'])
            if key in selected_sections:
                continue
            results.append(chunk)
            selected_sections.add(key)
            if len(results) == limit:
                break
        return results
