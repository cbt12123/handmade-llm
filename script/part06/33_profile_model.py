"""Profiler evidence that the actual model executes the course GPU kernel."""
import json
import torch
from common import OUT,save,require_cuda
from inference import load,encode,decode
from model_adapter import fused_qwen_norms


@torch.inference_mode()
def main():
    require_cuda();tokenizer,model=load()
    ids=encode(tokenizer,'请用一句话解释梯度下降。')
    original={name:module.forward for name,module in model.named_modules() if type(module).__name__=='Qwen2RMSNorm'}
    # Make sure exceptions restore methods, not only successful requests.
    try:
        with fused_qwen_norms(model):raise RuntimeError('injected cleanup check')
    except RuntimeError:
        pass
    assert all(model.get_submodule(name).forward==fn for name,fn in original.items())
    outputs={}
    for enabled in [False,True]:
        label='fused' if enabled else 'baseline';counter={}
        with fused_qwen_norms(model,enabled,counter) as patch:
            decode(model,ids,steps=4);torch.cuda.synchronize();counter.clear()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
                decode(model,ids,steps=4);torch.cuda.synchronize()
        events=prof.events()
        cuda_events=[event for event in events if str(event.device_type).endswith('CUDA')]
        kernel_counts={}
        for event in cuda_events:kernel_counts[event.name]=kernel_counts.get(event.name,0)+1
        averages=prof.key_averages()
        top=sorted(averages,key=lambda event:event.self_device_time_total,reverse=True)[:20]
        summary={'patched_modules':patch['patched_modules'],'wrapper_counter':counter,
                 'cuda_event_count':len(cuda_events),
                 'course_rms_kernel_count':sum(count for name,count in kernel_counts.items() if '_rms' in name),
                 'cuda_kernel_counts':kernel_counts,
                 'cpu_mean_operator_count':sum(e.count for e in averages if e.key=='aten::mean'),
                 'top_device_events':[{'name':e.key,'count':e.count,'self_device_us':e.self_device_time_total} for e in top]}
        OUT.mkdir(parents=True,exist_ok=True);prof.export_chrome_trace(str(OUT/f'trace-{label}.json'))
        outputs[label]=summary
        print(label,summary['cuda_event_count'],summary['course_rms_kernel_count'],flush=True)
    expected=(2*model.config.num_hidden_layers+1)*4
    assert outputs['fused']['patched_modules']==2*model.config.num_hidden_layers+1
    assert outputs['fused']['wrapper_counter']['calls']==expected
    assert outputs['fused']['wrapper_counter'].get('fallbacks',0)==0
    assert outputs['fused']['course_rms_kernel_count']==expected
    save('33_profile.json',{'generated_steps':4,'expected_rms_calls':expected,'results':outputs,
                           'methods_restored_after_exception':True,
                           'scope':'Real model profiler evidence only; instrumentation timings are not benchmark results.'})


if __name__=='__main__':main()
