"""Sampling arithmetic and equivalent cached/uncached real-model decoding."""
import time
import numpy as np
import torch
from _common import load_model, encode_messages, save_report


def softmax(logits, temperature=1.0):
    if temperature <= 0:
        raise ValueError("temperature must be positive for sampling")
    shifted = logits/temperature
    shifted = shifted-shifted.max()
    probabilities = np.exp(shifted)
    return probabilities/probabilities.sum()


def nucleus(probabilities, p):
    if not 0 < p <= 1:
        raise ValueError("p must be in (0,1]")
    order = np.argsort(-probabilities)
    # Keep the first token that crosses p as well as all earlier tokens.
    count = min(int(np.searchsorted(np.cumsum(probabilities[order]), p))+1, len(order))
    selected = order[:count]
    filtered = np.zeros_like(probabilities)
    filtered[selected] = probabilities[selected]
    return filtered/filtered.sum()


def synchronize(model):
    if model.device.type == "cuda":
        torch.cuda.synchronize()


def compare_cache(model, ids, steps=8):
    full, incremental, cache = ids.clone(), ids.clone(), None
    errors, predictions, full_times, cache_times = [], [], [], []
    for step in range(steps):
        synchronize(model)
        start = time.perf_counter()
        reference = model(input_ids=full, attention_mask=torch.ones_like(full), use_cache=False).logits[:, -1]
        synchronize(model)
        full_times.append(time.perf_counter()-start)
        synchronize(model)
        start = time.perf_counter()
        output = model(input_ids=incremental, attention_mask=torch.ones_like(full),
                       past_key_values=cache, use_cache=True)
        synchronize(model)
        cache_times.append(time.perf_counter()-start)
        cached_logits, cache = output.logits[:, -1], output.past_key_values
        error = float((reference-cached_logits).abs().max())
        assert torch.allclose(reference, cached_logits, atol=3e-4, rtol=3e-4), error
        next_id = reference.argmax(dim=-1, keepdim=True)
        assert torch.equal(next_id, cached_logits.argmax(dim=-1, keepdim=True))
        errors.append(error); predictions.append(int(next_id[0,0]))
        full = torch.cat((full, next_id), dim=1)
        incremental = next_id
    return {"steps":steps, "max_logit_errors":errors, "next_token_ids":predictions,
            "full_prefix_seconds":full_times, "cache_seconds":cache_times,
            "prefill_seconds":cache_times[0], "cached_decode_seconds":cache_times[1:],
            "full_prefix_decode_seconds":full_times[1:],
            "cache_sequence_length":int(cache.get_seq_length()),
            "expected_cache_sequence_length":int(full.shape[1]-1),
            "note":"Fixed-step derivative-free equivalence experiment; does not stop on EOS."}


def main():
    toy = np.array([2., 1., 0., -1.])
    sampling = {"logits":toy.tolist(), "temperature_1":softmax(toy).tolist(),
                "temperature_half":softmax(toy, .5).tolist(),
                "top_p_half":nucleus(softmax(toy), .5).tolist()}
    model, tokenizer, _ = load_model()
    messages = [{"role":"user", "content":"请用一句话解释梯度下降。"}]
    ids = encode_messages(tokenizer, messages, model.device).input_ids
    with torch.inference_mode():
        model(input_ids=ids[:, :2], use_cache=False)  # warmup, outside measurements
        comparison = compare_cache(model, ids)
    assert comparison["cache_sequence_length"] == comparison["expected_cache_sequence_length"]
    comparison["text"] = tokenizer.decode(comparison["next_token_ids"], skip_special_tokens=True)
    save_report("16_cache.json", {"sampling":sampling, "comparison":comparison,
                                  "device":str(model.device), "attention_implementation":"eager"})
    print("maximum logit difference:", max(comparison["max_logit_errors"]))


if __name__ == "__main__":
    main()
