"""Paths, local-only loading, and generation shared by Part IV."""
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "part04"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def model_path():
    # No author-specific drive path is hidden in this default.
    path = Path(os.environ.get("LLM_MODEL_PATH", str(ROOT / "models" / "qwen2.5-1.5b-instruct")))
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError("Set LLM_MODEL_PATH to a complete local model directory; see Part IV README.")
    return path


def save_report(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(OUT / name)


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path(), local_files_only=True, trust_remote_code=False)


def load_model():
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(int(os.environ.get("LLM_CPU_THREADS", "4")))
    torch.manual_seed(42)
    requested = os.environ.get("LLM_DEVICE", "auto")
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("LLM_DEVICE must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in the current PyTorch installation")
    device = "cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path(), local_files_only=True, trust_remote_code=False,
        dtype=dtype, attn_implementation="eager")
    model.to(device).eval()
    return model, load_tokenizer(), time.perf_counter()-start


def encode_messages(tokenizer, messages, device="cpu"):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, add_special_tokens=False, return_tensors="pt").to(device)


def generate(model, tokenizer, messages, max_new_tokens=64, do_sample=False, use_cache=True):
    import torch
    inputs = encode_messages(tokenizer, messages, model.device)
    kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample, "use_cache": use_cache,
              "pad_token_id": tokenizer.eos_token_id}
    if do_sample:
        kwargs.update(temperature=0.7, top_p=0.9)
    else:
        kwargs.update(temperature=1.0, top_p=1.0, top_k=50)
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(**inputs, **kwargs)
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    new_ids = outputs[0, inputs.input_ids.shape[1]:]
    return {"text": tokenizer.decode(new_ids, skip_special_tokens=True),
            "input_tokens": int(inputs.input_ids.shape[1]), "output_tokens": int(new_ids.numel()),
            "generation_seconds": elapsed, "token_ids": new_ids.tolist()}
