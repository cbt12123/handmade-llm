"""Reuse the read-only WSL inspection helper; never changes Part IV's report."""
from pathlib import Path
import importlib.metadata as metadata
import json
import platform
import sys
from record_part04_container import wsl

ROOT = Path(__file__).resolve().parents[1]


def main():
    name='handmade-llm-part05-vllm'
    inspected=json.loads(wsl(['docker','inspect',name]))[0]
    code='''import json,platform,importlib.metadata as m,torch
print(json.dumps({'python':platform.python_version(),'packages':{n:m.version(n) for n in ['torch','transformers','vllm','requests','numpy']},'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0)}))
'''
    packages=json.loads(wsl(['docker','exec','-i',name,'python3','-'],code))
    report={'host_python':platform.python_version(),
            'host_packages':{n:metadata.version(n) for n in ['requests','numpy','matplotlib']},
            'container_name':name,'image_tag':inspected['Config']['Image'],'image_id':inspected['Image'],
            'command':inspected['Config']['Cmd'],'running_at_verification':inspected['State']['Running'],
            'model_mount_read_only':any(m['Destination']=='/models/qwen' and not m['RW'] for m in inspected['Mounts']),
            'container':packages,'agent_backend':'vllm structured JSON actions',
            'optional_semantic_embedding_model_verified':False}
    assert packages['cuda_available'] and report['model_mount_read_only']
    output=ROOT/'outputs'/'part05'/'environment.json'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(output)


if __name__=='__main__':
    main()
