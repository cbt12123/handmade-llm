"""An explicit optional download; inference scripts never download implicitly."""
from pathlib import Path
from huggingface_hub import snapshot_download
from _common import ROOT, MODEL_ID


if __name__ == "__main__":
    destination = ROOT / "models" / "qwen2.5-1.5b-instruct"
    snapshot_download(MODEL_ID, local_dir=destination,
                      allow_patterns=["*.json", "*.txt", "*.safetensors", "LICENSE", "README.md"])
    print(Path(destination).resolve())
