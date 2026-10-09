"""Record a running teaching container's image and actual CUDA dependencies."""
from pathlib import Path
import argparse
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def wsl(arguments, code=None):
    completed = subprocess.run(["wsl","-d","Ubuntu","--exec",*arguments],
                               input=code.encode("utf-8") if code else None,capture_output=True,check=True)
    # Docker emits UTF-8 JSON; WSL warnings on stderr may use a Windows encoding.
    return completed.stdout.decode("utf-8")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--name",default="handmade-llm-part04-vllm")
    args=parser.parse_args()
    inspected=json.loads(wsl(["docker","inspect",args.name]))[0]
    code='''import json,platform,importlib.metadata as m,torch
names=["torch","transformers","vllm","fastapi","uvicorn","requests","numpy"]
print(json.dumps({"python":platform.python_version(),"packages":{n:m.version(n) for n in names},"torch_runtime_version":torch.__version__,"cuda_runtime":torch.version.cuda,"cuda_available":torch.cuda.is_available(),"gpu":torch.cuda.get_device_name(0),"gpu_total_memory_bytes":torch.cuda.get_device_properties(0).total_memory}))
'''
    environment=json.loads(wsl(["docker","exec","-i",args.name,"python3","-"],code))
    result={"container_name":args.name,"image_tag":inspected["Config"]["Image"],
            "image_id":inspected["Image"],"command":inspected["Config"]["Cmd"],
            "container_running_at_verification":inspected["State"]["Running"],
            "model_mount_read_only":any(m["Destination"]=="/models/qwen" and not m["RW"] for m in inspected["Mounts"]),
            "environment":environment}
    assert environment["cuda_available"] and result["model_mount_read_only"]
    output=ROOT/"outputs/part04/18_container.json"
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(output)


if __name__=="__main__":main()
