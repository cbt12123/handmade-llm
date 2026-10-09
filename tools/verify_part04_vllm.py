"""Verify an already started vLLM service; never stops unrelated containers."""
from pathlib import Path
import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "part04"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--url",default="http://127.0.0.1:8001")
    args=parser.parse_args()
    session=requests.Session();session.trust_env=False
    deadline=time.monotonic()+300
    print("Waiting for the already-started GPU service...",flush=True)
    while True:
        try:
            response=session.get(args.url+"/health",timeout=(2,2))
            if response.ok:break
        except requests.RequestException:pass
        if time.monotonic()>deadline:raise TimeoutError("GPU service unavailable; inspect container logs")
        time.sleep(1)
    for name,extra in [("18_vllm_client.py",[]),("19_evaluate.py",["--backend","vllm"])]:
        print("Running",name,flush=True)
        subprocess.run([sys.executable,"-X","utf8",str(ROOT/"script/part04"/name),"--url",args.url,*extra],check=True,cwd=ROOT)
    def request(number):
        local=requests.Session();local.trust_env=False
        response=local.post(args.url+"/v1/chat/completions",json={"model":"tutorial-qwen",
            "messages":[{"role":"user","content":f"计算{number}+1，只输出整数。"}],
            "temperature":0,"max_tokens":16},timeout=(5,120))
        response.raise_for_status()
        text=response.json()["choices"][0]["message"]["content"].strip()
        return {"input":number,"output":text,"correct":text==str(number+1)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        concurrent_results=list(pool.map(request,[1,2,3,4]))
    assert all(r["correct"] for r in concurrent_results),concurrent_results
    http=json.loads((OUT/"18_vllm_http.json").read_text(encoding="utf-8"))
    assert http["stream"]["same_as_nonstream"] and http["structured"]["matches_expected"]
    result={"health_status":200,"concurrent_requests":concurrent_results,
            "stream_matches_nonstream":True,"structured_json_matches_expected":True,
            "evaluation_report":"19_evaluation_vllm.json","scope":"Functional checks, not a throughput benchmark"}
    (OUT/"verification_vllm.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
