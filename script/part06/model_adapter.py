"""Temporary, reversible replacement of real Qwen RMSNorm forward methods."""
from contextlib import contextmanager
import types
import torch
from kernels import rms_norm,rms_supported


@contextmanager
def fused_qwen_norms(model,enabled=True,counter=None):
    originals=[]
    if enabled:
        for name,module in model.named_modules():
            if type(module).__name__=='Qwen2RMSNorm':
                original=module.forward

                def forward(self,hidden_states,_original=original):
                    if counter is not None:
                        counter['calls']=counter.get('calls',0)+1
                    if torch.is_grad_enabled() or not rms_supported(hidden_states,self.weight):
                        if counter is not None:counter['fallbacks']=counter.get('fallbacks',0)+1
                        return _original(hidden_states)
                    return rms_norm(hidden_states,self.weight,self.variance_epsilon)

                originals.append((module,original,name))
                module.forward=types.MethodType(forward,module)
        if not originals:
            raise ValueError('未找到 Qwen2RMSNorm，不能声称已经接入')
    try:
        yield {'patched_modules':len(originals),'names':[name for _,_,name in originals]}
    finally:
        for module,original,_ in originals:
            module.forward=original
