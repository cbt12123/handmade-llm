"""Local teaching deployment with the real fused operator in generation."""
from contextlib import asynccontextmanager
import threading
import time
from typing import Literal
import torch
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from inference import load,encode,decode
from model_adapter import fused_qwen_norms

gate=threading.Lock()


@asynccontextmanager
async def lifespan(app):
    tokenizer,model=load()
    app.state.tokenizer,app.state.model=tokenizer,model
    with torch.inference_mode():
        with fused_qwen_norms(model):decode(model,encode(tokenizer,'你好'),steps=2)
    yield


app=FastAPI(title='Course Qwen operator deployment',lifespan=lifespan)


class Request(BaseModel):
    prompt:str=Field(min_length=1,max_length=1200)
    backend:Literal['baseline','fused']='fused'
    max_new_tokens:int=Field(default=32,ge=1,le=128)


@app.get('/health')
def health():
    return {'status':'ok','model_type':app.state.model.config.model_type,'default_backend':'fused',
            'scope':'Batch-one local teaching service, SDPA and greedy, no production engine speed claim'}


@app.post('/generate')
def generate(body:Request):
    if not gate.acquire(blocking=False):raise HTTPException(429,'服务正忙')
    try:
        with torch.inference_mode():
            ids=encode(app.state.tokenizer,body.prompt)
            if ids.shape[1]+body.max_new_tokens>2048:raise HTTPException(413,'上下文预算超出2048')
            counter={}
            torch.cuda.synchronize();start=time.perf_counter()
            with fused_qwen_norms(app.state.model,body.backend=='fused',counter) as patch:
                output=decode(app.state.model,ids,steps=body.max_new_tokens,stop_eos=True)
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            tokens=output['ids'][0].cpu().tolist()
            eos=app.state.model.generation_config.eos_token_id
            eos=eos if isinstance(eos,list) else [eos]
            return {'text':app.state.tokenizer.decode(tokens,skip_special_tokens=True),'ids':tokens,
                    'backend':body.backend,'input_tokens':ids.shape[1],'output_tokens':len(tokens),
                    'finish_reason':'eos' if tokens[-1] in eos else 'length',
                    'generation_seconds':seconds,'patched_modules':patch['patched_modules'],
                    'custom_kernel_calls':counter.get('calls',0),'fallbacks':counter.get('fallbacks',0)}
    finally:
        gate.release()
