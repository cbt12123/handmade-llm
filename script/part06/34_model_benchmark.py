"""Paired actual inference: same model, dtype, attention, cache and selector."""
import argparse
import hashlib
import json
import statistics
import time
import torch
from common import require_cuda,save,model_path
from inference import load,encode,decode
from model_adapter import fused_qwen_norms


def timed(fn):
    torch.cuda.synchronize();start=time.perf_counter()
    value=fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-start)*1000,value


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repeats',type=int,default=9)
    parser.add_argument('--steps',type=int,default=32);args=parser.parse_args()
    if args.repeats<3 or args.steps<1:parser.error('至少3轮，每轮至少1个token')
    require_cuda();tokenizer,model=load()
    prompts=['用一句话解释什么是矩阵。','梯度下降中学习率有什么作用？','公平硬币抛两次，两次正面的概率是多少？']
    verification=[]
    for prompt in prompts:
        ids=encode(tokenizer,prompt)
        base=decode(model,ids,steps=8,capture=True)
        with fused_qwen_norms(model) as patch:
            optimized=decode(model,ids,steps=8,capture=True)
        differences=[float((a-b).abs().max()) for a,b in zip(base['logits'],optimized['logits'])]
        same=torch.equal(base['ids'],optimized['ids'])
        verification.append({'prompt':prompt,'input_tokens':ids.shape[1],'same_ids':same,
            'baseline_ids':base['ids'].cpu().tolist(),'fused_ids':optimized['ids'].cpu().tolist(),
            'baseline_text':tokenizer.decode(base['ids'][0]),'fused_text':tokenizer.decode(optimized['ids'][0]),
            'max_logit_errors':differences,'patch':patch})
        if not same:raise AssertionError('生成ID不一致，不能按等价优化报告收益')
        for a,b in zip(base['logits'],optimized['logits']):
            torch.testing.assert_close(a,b,atol=.06,rtol=.01)
    seed=encode(tokenizer,'梯度下降沿负梯度方向更新参数。')
    measurements=[]
    for length in [128,512,1024]:
        ids=seed.repeat(1,(length+seed.shape[1]-1)//seed.shape[1])[:,:length].contiguous()
        # Compile/warm both routes before any measurement. No model load timed.
        for enabled in [False,True]:
            with fused_qwen_norms(model,enabled):decode(model,ids,steps=4)
        raw={'baseline':[],'fused':[]}
        prefill={'baseline':[],'fused':[]}
        generated_ids={}
        for repeat in range(args.repeats):
            order=[False,True] if repeat%2==0 else [True,False]
            for enabled in order:
                name='fused' if enabled else 'baseline'
                with fused_qwen_norms(model,enabled):
                    latency,output=timed(lambda:decode(model,ids,steps=args.steps))
                    prefix_latency,_=timed(lambda:model(input_ids=ids,use_cache=True,logits_to_keep=1))
                raw[name].append(latency);prefill[name].append(prefix_latency)
                token_ids=output['ids'].cpu().tolist()
                if name in generated_ids and token_ids != generated_ids[name]:
                    raise AssertionError('同一路径重复生成不稳定')
                generated_ids[name]=token_ids
            if generated_ids['baseline'] != generated_ids['fused']:
                raise AssertionError('基准长度的完整生成ID不一致，不能按等价优化交付')
        total_speed=statistics.median(raw['baseline'])/statistics.median(raw['fused'])
        prefix_speed=statistics.median(prefill['baseline'])/statistics.median(prefill['fused'])
        row={'input_tokens':length,'generated_tokens_fixed':args.steps,'total_ms':raw,'prefill_only_ms':prefill,
             'total_median_speedup':total_speed,'prefill_median_speedup':prefix_speed,
             'baseline_total_median_ms':statistics.median(raw['baseline']),
             'fused_total_median_ms':statistics.median(raw['fused']),
             'same_full_generated_ids':True,'generated_ids':generated_ids}
        measurements.append(row);print(length,'total',round(total_speed,3),'prefill',round(prefix_speed,3),flush=True)
        save('34_real_inference.json',{'model_type':model.config.model_type,'hidden_size':model.config.hidden_size,
             'num_layers':model.config.num_hidden_layers,'dtype':'float16','attention':'sdpa','batch_size':1,
             'logits_to_keep':1,'selector':'argmax','fixed_steps_ignore_eos_for_timing':True,
             'repeats':args.repeats,'verification':verification,'measurements':measurements,
             'config_sha256':hashlib.sha256((model_path()/'config.json').read_bytes()).hexdigest(),
             'scope':'Real Qwen greedy GPU inference, original HF eager RMSNorm vs course Triton replacement; no vLLM or full-model torch.compile speed claim.'})
    # A separate prefill-only shortcut: avoid logits for unused prefix positions.
    length=512;ids=seed.repeat(1,(length+seed.shape[1]-1)//seed.shape[1])[:,:length].contiguous()
    full=model(input_ids=ids,use_cache=True,logits_to_keep=0).logits
    tail=model(input_ids=ids,use_cache=True,logits_to_keep=1).logits
    torch.testing.assert_close(full[:,-1,:],tail[:,0,:],atol=.04,rtol=.01)
    results={'all_positions':[],'last_position':[]}
    for repeat in range(args.repeats):
        for keep in ([0,1] if repeat%2==0 else [1,0]):
            ms,_=timed(lambda:model(input_ids=ids,use_cache=True,logits_to_keep=keep))
            results['all_positions' if keep==0 else 'last_position'].append(ms)
    save('32_last_logits.json',{'input_tokens':length,'vocabulary':model.config.vocab_size,
         'all_positions_shape':list(full.shape),'last_position_shape':list(tail.shape),
         'last_logits_max_error':float((full[:,-1,:]-tail[:,0,:]).abs().max()),
         'last_token_same':torch.equal(full[:,-1,:].argmax(-1),tail[:,0,:].argmax(-1)),
         'timings_ms':results,'median_speedup':statistics.median(results['all_positions'])/statistics.median(results['last_position']),
         'scope':'Separate prefill workflow shortcut; both RMSNorm inference variants above already use logits_to_keep=1.'})


if __name__=='__main__':main()
