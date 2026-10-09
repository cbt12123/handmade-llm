"""A real HTTP client, with explicit network timeouts."""
import argparse
import requests
from _common import save_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    session = requests.Session()
    session.trust_env = False  # Local requests should not go through a global proxy.
    health = session.get(args.url+"/health", timeout=(5,30))
    health.raise_for_status()
    response = session.post(args.url+"/generate", json={"messages":[
        {"role":"system","content":"请用中文简短回答。"},
        {"role":"user","content":"用一句话解释什么是矩阵。"}], "max_new_tokens":64}, timeout=(5,180))
    response.raise_for_status()
    data = response.json()
    assert data["input_tokens"] > 0 and 0 < data["output_tokens"] <= 64
    print(data["text"])
    save_report("18_http.json", {"health":health.json(), "response":data})


if __name__ == "__main__":
    main()
