"""Compile native examples in the existing CUDA devel image, no PyTorch needed."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
from common import ROOT,OUT,save


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--arch',default='sm_89')
    parser.add_argument('--sanitize',action='store_true')
    args=parser.parse_args()
    if not args.arch.startswith('sm_') or not args.arch[3:].isdigit():
        parser.error('架构格式为 sm_89 等，请查GPU实际计算能力')
    if not shutil.which('nvcc'):
        raise RuntimeError('需要已有CUDA devel环境中的nvcc')
    build=OUT/'build';build.mkdir(parents=True,exist_ok=True)
    report={'arch':args.arch,'examples':{},'scope':'Real compiled native CUDA and cuBLAS; not CPU simulation'}
    for name in ['vector_add','rms_norm','matmul']:
        source=ROOT/'assets'/'part06'/'cuda'/f'{name}.cu'
        target=build/name
        command=['nvcc','-O3',f'-arch={args.arch}',str(source),'-o',str(target)]
        if name=='matmul':command.append('-lcublas')
        subprocess.run(command,check=True)
        result=json.loads(subprocess.check_output([str(target)],text=True))
        report['examples'][name]=result
        if name=='matmul':
            report['examples']['matmul_tail']=json.loads(subprocess.check_output([str(target),'small'],text=True))
        if args.sanitize:
            if not shutil.which('compute-sanitizer'):
                raise RuntimeError('请求sanitize但工具不存在')
            check=subprocess.run(['compute-sanitizer','--tool','memcheck','--error-exitcode','99',str(target),*(['small'] if name=='matmul' else [])],
                                 check=True,capture_output=True,text=True)
            report['examples'][name]['sanitizer_log']=check.stdout+check.stderr
    save('29_native_cuda.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
