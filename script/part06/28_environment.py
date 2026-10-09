"""Inspect actual GPU/compiler packages; save no credentials or private paths."""
import importlib.metadata as metadata
import platform
import shutil
import subprocess
from common import require_cuda,save


def main():
    torch=require_cuda();properties=torch.cuda.get_device_properties(0)
    nvcc=shutil.which('nvcc')
    result={'python':platform.python_version(),'packages':{n:metadata.version(n) for n in ['torch','triton','transformers','numpy']},
            'torch_cuda_runtime':torch.version.cuda,'gpu':properties.name,
            'compute_capability':[properties.major,properties.minor],
            'gpu_memory_bytes':properties.total_memory,'sm_count':properties.multi_processor_count,
            'nvcc_available':nvcc is not None,
            'nvcc_version':subprocess.check_output([nvcc,'--version'],text=True).strip() if nvcc else None,
            'compute_sanitizer_available':shutil.which('compute-sanitizer') is not None}
    save('28_environment.json',result);print(result)


if __name__=='__main__':
    main()
