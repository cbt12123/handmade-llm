"""Verify real local inference, HTTP boundary behavior, and chapter examples."""
from pathlib import Path
import concurrent.futures
import contextlib
import gc
import io
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import requests
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "part04"
PART = ROOT / "第四部分-大模型与部署"


def run(name):
    print("Running", name, flush=True)
    subprocess.run([sys.executable, "-X", "utf8", str(ROOT/"script"/"part04"/name)], cwd=ROOT, check=True)


def verify_http():
    # Never kill or send test requests to an unrelated server already on this port.
    with socket.socket() as probe:
        try: probe.bind(("127.0.0.1",8000))
        except OSError as error: raise RuntimeError("Port 8000 is occupied; stop your own service before verification") from error
    session = requests.Session(); session.trust_env = False
    checks = {}
    log_path = OUT/"server-verification.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable,"-X","utf8",str(ROOT/"script/part04/18_server.py")],
                                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+120
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Server exited; inspect outputs/part04/server-verification.log")
                try:
                    response = session.get("http://127.0.0.1:8000/health",timeout=2)
                    if response.ok: break
                except requests.RequestException: pass
                if time.monotonic() > deadline: raise TimeoutError("Model service startup timed out")
                time.sleep(.25)
            run("18_client.py")
            def post(payload):
                return session.post("http://127.0.0.1:8000/generate",json=payload,timeout=(5,180))
            invalid = post({"messages":[{"role":"tool","content":"x"}]})
            assert invalid.status_code == 422
            checks["invalid_role"] = invalid.status_code
            invalid_limit = post({"messages":[{"role":"user","content":"x"}],"max_new_tokens":129})
            assert invalid_limit.status_code == 422
            checks["invalid_output_limit"] = invalid_limit.status_code
            too_long = post({"messages":[{"role":"user","content":"你好"*1000}],"max_new_tokens":64})
            assert too_long.status_code == 413
            checks["context_limit"] = too_long.status_code
            # Force a longer request, wait until its generation owns the admission lock.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(post,{"messages":[{"role":"user","content":"列出从1到100的整数，用逗号分隔，不要省略。"}],"max_new_tokens":128})
                deadline = time.monotonic()+10
                while not session.get("http://127.0.0.1:8000/health",timeout=2).json()["busy"]:
                    if future.done() or time.monotonic()>deadline: raise AssertionError("Could not observe active generation")
                    time.sleep(.02)
                busy = post({"messages":[{"role":"user","content":"你好"}],"max_new_tokens":8})
                assert busy.status_code == 429
                checks["concurrent_request"] = busy.status_code
                assert future.result().status_code == 200
            # The gate must be released after both validation failures and successful generation.
            assert not session.get("http://127.0.0.1:8000/health",timeout=2).json()["busy"]
            run("19_evaluate.py")
        finally:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
    (OUT/"18_http_boundaries.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    return checks


def verify_failure_cleanup():
    sys.path.insert(0,str(ROOT/"script/part04"))
    spec=importlib.util.spec_from_file_location("part04_server_checks",ROOT/"script/part04/18_server.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.app.state.tokenizer=object()
    request=module.GenerateRequest(messages=[{"role":"user","content":"test"}])
    with patch.object(module,"encode_messages",side_effect=RuntimeError("injected failure")):
        with patch.object(module.logger,"exception"):
            try:module.generate_text(request)
            except module.HTTPException as error:
                assert error.status_code==500 and error.detail["request_id"]
            else:raise AssertionError("Injected model failure should return 500")
    assert not module.gate.locked()
    return True


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    checks = {}
    if "--check-only" not in sys.argv:
        for script in ("14_model_structure.py","15_local_inference.py","16_generation_cache.py","17_memory_quantization.py"):
            run(script)
        checks = verify_http()
        run("19_release_manifest.py")
    failure_cleanup=verify_failure_cleanup()
    count, links = 0, 0
    for doc in sorted(PART.glob("*.md")):
        text = doc.read_text(encoding="utf-8")
        assert text.count("```")%2 == 0, doc
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
            if "://" in target or target.startswith("#"): continue
            assert (doc.parent/target.split("#")[0]).exists(),(doc.name,target)
            links += 1
        namespace = {"__name__":"__main__"}
        # Examples with local model files need LLM_MODEL_PATH inherited from the caller.
        for snippet in re.findall(r'```python\s*\n(.*?)\n```',text,re.S):
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(snippet,str(doc),"exec"),namespace)
            count += 1
        namespace.clear(); gc.collect()
    reports = {p.stem:json.loads(p.read_text(encoding="utf-8")) for p in OUT.glob("*.json") if p.stem!="verification"}
    if not checks and "18_http_boundaries" in reports:
        checks = reports["18_http_boundaries"]
    if reports:
        assert reports["14_structure"]["weights_loaded"] is False
        assert all(t["decoded"] == t["text"] for t in reports["15_inference"]["token_examples"])
        cache = reports["16_cache"]["comparison"]
        assert max(cache["max_logit_errors"]) < 3e-4
        assert cache["cache_sequence_length"] == cache["expected_cache_sequence_length"]
        assert reports["17_memory"]["lora_merge_max_error"] < 1e-10
        assert reports["19_evaluation"]["count"] == 8
    result = {"python":sys.version,"python_examples_executed":count,"local_links_checked":links,
              "http_boundary_checks":checks,"report_files":sorted(reports),
              "injected_failure_returns_request_id_and_releases_lock":failure_cleanup,
              "mode":"existing reports and examples" if "--check-only" in sys.argv else "real local model and real HTTP"}
    (OUT/"verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
