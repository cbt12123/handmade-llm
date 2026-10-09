"""Real local Qwen GPU load and a bounded batch-one cached greedy decoder."""
import torch
from functools import lru_cache
from transformers import AutoModelForCausalLM,AutoTokenizer
from common import model_path


@lru_cache(maxsize=1)
def load():
    path=str(model_path())
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,trust_remote_code=False,
                                              dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval()
    return tokenizer,model


def encode(tokenizer,text):
    prompt=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True)
    return tokenizer(prompt,return_tensors='pt',add_special_tokens=False)['input_ids'].to('cuda')


@torch.inference_mode()
def decode(model,ids,steps=32,keep=1,capture=False,stop_eos=False):
    cache=None;current=ids;generated=[];logits=[]
    eos=model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])
    for _ in range(steps):
        result=model(input_ids=current,past_key_values=cache,use_cache=True,logits_to_keep=keep)
        final=result.logits[:,-1,:]
        selected=final.argmax(-1,keepdim=True)
        generated.append(selected)
        if capture:logits.append(final.detach().float().cpu())
        cache=result.past_key_values;current=selected
        if stop_eos and int(selected.item()) in eos:break
    return {'ids':torch.cat(generated,dim=1),'logits':logits}
