"""Inspect actual tokenizer behavior, then run one local model generation."""
from _common import load_model, encode_messages, save_report
import torch


def main():
    model, tokenizer, load_seconds = load_model()
    token_examples = []
    for text in ("Python", "我正在学习Python。", "2 + 3 = 5"):
        ids = tokenizer.encode(text, add_special_tokens=False)
        token_examples.append({"text":text, "characters":len(text), "ids":ids,
                               "pieces":tokenizer.convert_ids_to_tokens(ids),
                               "decoded":tokenizer.decode(ids)})
        assert tokenizer.decode(ids) == text
    messages = [{"role":"system", "content":"请用中文简短回答。"},
                {"role":"user", "content":"用一句话解释什么是矩阵。"}]
    inputs = encode_messages(tokenizer, messages, model.device)
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False).logits[0, -1].float()
    probabilities = logits.softmax(dim=-1)
    values, ids = probabilities.topk(5)
    from _common import generate
    result = generate(model, tokenizer, messages)
    data = {"device":str(model.device), "dtype":str(next(model.parameters()).dtype),
            "load_seconds":load_seconds, "actual_parameters":sum(p.numel() for p in model.parameters()),
            "token_examples":token_examples,
            "chat_template_text":tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            "next_token_top5":[{"id":int(i), "piece":tokenizer.decode([int(i)]), "probability":float(v)} for i,v in zip(ids,values)],
            "generation":result}
    print(result["text"])
    save_report("15_inference.json", data)


if __name__ == "__main__":
    main()
