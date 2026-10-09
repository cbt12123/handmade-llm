"""Evaluate exact answers and strict JSON through the deployed endpoint."""
import argparse
import hashlib
import json
import re
import time
import numpy as np
import requests
from _common import ROOT, save_report


def judge(case, text):
    stripped = text.strip()
    if case["category"] == "arithmetic":
        valid = bool(re.fullmatch(r"-?\d+", stripped))
        return {"format_valid":valid, "correct":valid and int(stripped) == case["expected"]}
    if case["category"] == "json":
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            return {"format_valid":False, "correct":False}
        valid = (isinstance(obj, dict) and set(obj) == {"name","age"}
                 and isinstance(obj["name"], str) and type(obj["age"]) is int)
        return {"format_valid":valid, "correct":valid and obj == case["expected"]}
    return {"format_valid":True, "correct":stripped == case["expected"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--backend", choices=["tutorial", "vllm"], default="tutorial")
    parser.add_argument("--model", default="tutorial-qwen")
    args = parser.parse_args()
    dataset = ROOT / "data" / "part04" / "evaluation.jsonl"
    cases = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    session = requests.Session(); session.trust_env = False
    # Warmup excluded from latency statistics.
    def payload(prompt, limit):
        messages = [{"role":"user","content":prompt}]
        if args.backend == "vllm":
            return {"model":args.model,"messages":messages,"max_tokens":limit,"temperature":0,"seed":42}
        return {"messages":messages,"max_new_tokens":limit}
    endpoint = args.url + ("/v1/chat/completions" if args.backend == "vllm" else "/generate")
    warmup = session.post(endpoint, json=payload("你好",8), timeout=(5,180))
    warmup.raise_for_status()
    results = []
    for case in cases:
        start = time.perf_counter()
        response = session.post(endpoint,json=payload(case["prompt"],64),timeout=(5,180))
        elapsed = time.perf_counter()-start
        response.raise_for_status()
        result = response.json()
        if args.backend == "vllm":
            result = {"text":result["choices"][0]["message"]["content"],
                      "input_tokens":result["usage"]["prompt_tokens"],
                      "output_tokens":result["usage"]["completion_tokens"],
                      "finish_reason":result["choices"][0]["finish_reason"]}
        results.append({"id":case["id"],"category":case["category"],"expected":case["expected"],
                        "output":result["text"],"http_seconds":elapsed,
                        "input_tokens":result["input_tokens"],"output_tokens":result["output_tokens"],
                        "finish_reason":result["finish_reason"],**judge(case,result["text"])})
    categories = {}
    for category in sorted({r["category"] for r in results}):
        subset = [r for r in results if r["category"] == category]
        categories[category] = {"count":len(subset),"correct":sum(r["correct"] for r in subset),
                                "format_valid":sum(r["format_valid"] for r in subset)}
    filename = "19_evaluation_vllm.json" if args.backend == "vllm" else "19_evaluation.json"
    save_report(filename, {"backend":args.backend,"dataset_sha256":hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "count":len(results),"correct":sum(r["correct"] for r in results),"categories":categories,
                "latency_p50_seconds":float(np.percentile([r["http_seconds"] for r in results],50)),
                "latency_p95_seconds":float(np.percentile([r["http_seconds"] for r in results],95)),
                "total_generated_tokens_per_second":sum(r["output_tokens"] for r in results)/sum(r["http_seconds"] for r in results),
                "results":results,"scope":"Eight authored teaching cases, sequential requests; not a general benchmark."})


if __name__ == "__main__":
    main()
