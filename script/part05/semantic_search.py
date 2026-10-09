"""Optional local-only embedding retrieval; no weights downloaded implicitly."""
import argparse
import hashlib
import json
from pathlib import Path
from common import OUT
from knowledge import Knowledge, digest


def model_fingerprint(path):
    hasher = hashlib.sha256()
    files = [p for p in sorted(path.rglob('*')) if p.is_file()]
    if not files:
        raise ValueError('嵌入模型目录为空')
    for file in files:
        hasher.update(file.relative_to(path).as_posix().encode())
        with file.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024), b''):
                hasher.update(block)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--query')
    args = parser.parse_args()
    model_path = Path(args.model_path).resolve()
    if not model_path.is_dir():
        parser.error('需要已有的本地 Sentence Transformers 模型目录')
    # Import only in this optional extension.
    import numpy as np
    from sentence_transformers import SentenceTransformer
    knowledge = Knowledge()
    fingerprint = model_fingerprint(model_path)
    model = SentenceTransformer(str(model_path), local_files_only=True, trust_remote_code=False, device='cpu')
    target, manifest_path = OUT/'semantic_vectors.npy', OUT/'semantic_manifest.json'
    if args.build:
        vectors = model.encode_document([c['text'] for c in knowledge.chunks], normalize_embeddings=True)
        np.save(target, vectors)
        manifest_path.write_text(json.dumps({'model_sha256':fingerprint,'index_sha256':digest(OUT/'course_index.json'),
                                             'dimension':vectors.shape[1]},indent=2),encoding='utf-8')
        print('Built', vectors.shape)
    if args.query:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['model_sha256'] != fingerprint or manifest['index_sha256'] != digest(OUT/'course_index.json'):
            raise ValueError('模型或知识索引变化，请重建向量')
        vectors = np.load(target, allow_pickle=False)
        query = model.encode_query([args.query], normalize_embeddings=True)[0]
        scores = vectors @ query
        print(json.dumps([{**knowledge.chunks[int(i)],'score':float(scores[i])}
                          for i in np.argsort(-scores)[:3]],ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
