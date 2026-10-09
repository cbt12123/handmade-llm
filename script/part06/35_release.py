"""Create a portable fingerprint list without weights, binaries or private paths."""
from pathlib import Path
import hashlib
import json
from common import ROOT,OUT,save


def main():
    required=['28_cost_model.json','28_environment.json','29_native_cuda.json',
              '31_correctness.json','31_microbenchmark.json','32_selection.json',
              '32_last_logits.json','33_profile.json','34_real_inference.json','35_http.json']
    for name in required:
        if not (OUT/name).is_file():raise FileNotFoundError(name)
    files=[*sorted((ROOT/'script/part06').glob('*.py')),
           *sorted((ROOT/'assets/part06/cuda').glob('*.cu')),
           ROOT/'tools/verify_part06_gpu.py',ROOT/'tools/draw_part06.py',
           *(OUT/name for name in required)]
    fingerprints={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    save('35_manifest.json',{'model_config_sha256':json.loads((OUT/'34_real_inference.json').read_text(encoding='utf-8'))['config_sha256'],
                            'sha256':fingerprints,'scope':'Source and observed reports; no weights or compiled binaries included.'})
    print('Recorded',len(fingerprints),'files')


if __name__=='__main__':main()
