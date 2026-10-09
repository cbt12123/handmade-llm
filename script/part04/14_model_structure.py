"""Inspect a real model configuration without loading its weights."""
import json
import importlib.metadata as metadata
import platform
from _common import model_path, save_report


def main():
    config = json.loads((model_path()/"config.json").read_text(encoding="utf-8"))
    d = config["hidden_size"]
    qh, kvh = config["num_attention_heads"], config["num_key_value_heads"]
    assert d % qh == 0 and qh % kvh == 0
    selected = {key: config[key] for key in ("model_type", "hidden_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "intermediate_size", "vocab_size",
                "max_position_embeddings", "tie_word_embeddings", "rope_theta")}
    selected.update(head_dim=d//qh, query_projection_shape=[d,d],
                    key_projection_shape=[d,kvh*(d//qh)],
                    embedding_parameters=config["vocab_size"]*d)
    print(json.dumps(selected, ensure_ascii=False, indent=2))
    packages = {}
    for name in ("torch", "transformers", "tokenizers", "huggingface-hub", "safetensors", "fastapi", "uvicorn", "requests", "pydantic", "numpy"):
        try: packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError: packages[name] = None
    save_report("14_structure.json", {"config":selected, "python":platform.python_version(), "packages":packages,
                                      "weights_loaded":False})


if __name__ == "__main__":
    main()
