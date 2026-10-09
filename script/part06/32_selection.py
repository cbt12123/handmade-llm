"""A scoped exact-algorithm shortcut: greedy needs argmax, not softmax."""
import torch
from common import require_cuda,bench,save


@torch.inference_mode()
def main():
    require_cuda()
    logits=torch.randn(1,151936,device='cuda',dtype=torch.float32)
    direct=logits.argmax(-1)
    via_probability=logits.softmax(-1).argmax(-1)
    assert torch.equal(direct,via_probability)
    result={'vocabulary_size':151936,'same_id_on_random_logits':True,
            'argmax':bench(lambda:logits.argmax(-1)),
            'softmax_then_argmax':bench(lambda:logits.softmax(-1).argmax(-1)),
            'scope':'Greedy selection shortcut only; model baseline already uses direct argmax. Not sampling or speculative decoding.'}
    # In exact arithmetic ordering is invariant, but FP rounding can merge scores.
    close=torch.tensor([[0.0,torch.finfo(torch.float32).eps/8]],device='cuda')
    result['rounding_counterexample']={'direct_id':int(close.argmax(-1).item()),
                                       'softmax_id':int(close.softmax(-1).argmax(-1).item()),
                                       'logits':close.cpu().tolist()}
    save('32_selection.json',result);print(result)


if __name__=='__main__':
    main()
