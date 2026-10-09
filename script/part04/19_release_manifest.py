"""Create a reproducible manifest without copying or uploading model weights."""
import hashlib
import json
from _common import model_path, ROOT, MODEL_ID, save_report


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    files = []
    for path in sorted(model_path().iterdir()):
        if path.is_file() and (path.suffix in (".json", ".txt", ".safetensors") or path.name == "LICENSE"):
            files.append({"name":path.name,"bytes":path.stat().st_size,"sha256":sha256(path)})
    config = json.loads((model_path()/"config.json").read_text(encoding="utf-8"))
    save_report("19_manifest.json", {"model_id":MODEL_ID,"model_type":config["model_type"],
                "files":files,"source":"https://huggingface.co/"+MODEL_ID,
                "license":"Apache-2.0; use the LICENSE in the model directory",
                "evaluation_sha256":sha256(ROOT/"data/part04/evaluation.jsonl"),
                "service":{"context_limit":512,"output_limit":128,"workers":1,"sampling":"greedy"}})


if __name__ == "__main__":
    main()
