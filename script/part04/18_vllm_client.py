"""Check actual vLLM HTTP, SSE streaming, and schema-constrained output."""
import argparse
import json
import time
import requests
from _common import save_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",default="http://127.0.0.1:8001")
    parser.add_argument("--model",default="tutorial-qwen")
    args = parser.parse_args()
    session=requests.Session();session.trust_env=False
    listing=session.get(args.url+"/v1/models",timeout=(5,30));listing.raise_for_status()
    assert args.model in [m["id"] for m in listing.json()["data"]]
    payload={"model":args.model,"messages":[{"role":"user","content":"用一句话解释什么是矩阵。"}],
             "max_tokens":64,"temperature":0,"seed":42}
    start=time.perf_counter()
    response=session.post(args.url+"/v1/chat/completions",json=payload,timeout=(5,180))
    response.raise_for_status()
    whole=response.json();whole_seconds=time.perf_counter()-start
    streaming={**payload,"stream":True,"stream_options":{"include_usage":True}}
    first_visible=None;parts=[];usage=None;finish=None;done=False
    start=time.perf_counter()
    with session.post(args.url+"/v1/chat/completions",json=streaming,stream=True,timeout=(5,180)) as stream:
        stream.raise_for_status()
        for line in stream.iter_lines(chunk_size=1):
            if not line.startswith(b"data: "):continue
            data=line[6:]
            if data == b"[DONE]":done=True;break
            event=json.loads(data)
            if event.get("usage"):usage=event["usage"]
            for choice in event.get("choices",[]):
                content=choice.get("delta",{}).get("content") or ""
                if content:
                    if first_visible is None:first_visible=time.perf_counter()-start
                    parts.append(content)
                if choice.get("finish_reason"):finish=choice["finish_reason"]
    stream_seconds=time.perf_counter()-start
    assert done and first_visible is not None and usage and finish
    assembled="".join(parts)
    assert assembled == whole["choices"][0]["message"]["content"]
    schema={"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},
            "required":["name","age"],"additionalProperties":False}
    constrained={**payload,"messages":[{"role":"user","content":"从句子提取姓名和年龄：王明今年20岁。只输出JSON对象，键为name和age。"}],
                 "structured_outputs":{"json":schema}}
    constrained_response=session.post(args.url+"/v1/chat/completions",json=constrained,timeout=(5,180))
    constrained_response.raise_for_status()
    raw=constrained_response.json()["choices"][0]["message"]["content"]
    parsed=json.loads(raw)
    assert set(parsed)=={"name","age"} and type(parsed["age"]) is int
    save_report("18_vllm_http.json",{"models":listing.json(),"response":whole,"http_seconds":whole_seconds,
                "stream":{"assembled_text":assembled,"same_as_nonstream":True,"first_visible_seconds":first_visible,
                          "total_seconds":stream_seconds,"usage":usage,"finish_reason":finish,"done_received":done},
                "structured":{"raw":raw,"parsed":parsed,"matches_expected":parsed=={"name":"王明","age":20}},
                "scope":"Single request checks; no claim about concurrency throughput or exact server-side TTFT."})
    print(assembled)


if __name__ == "__main__":
    main()
