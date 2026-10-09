"""Real Triton correctness and fair synthetic microbenchmarks, not model speed."""
import json
import torch
from common import require_cuda,rms_reference,bench,save
from kernels import vector_add,rms_norm,softmax


@torch.inference_mode()
def main():
    require_cuda()
    checks,errors={},[]
    for size in [0,1,257,1003,1048579]:
        a=torch.randn(size,device='cuda');b=torch.randn_like(a)
        torch.testing.assert_close(vector_add(a,b),a+b,rtol=0,atol=0)
    checks['vector_add_odd_tail_and_empty']=True
    for dtype,tol in [(torch.float32,2e-5),(torch.float16,3e-3),(torch.bfloat16,3e-2)]:
        for shape in [(1,1),(1,257),(17,1537),(1,1,1536),(0,1536),(32,4096)]:
            x=torch.randn(shape,device='cuda',dtype=dtype)
            weight=torch.randn(shape[-1],device='cuda',dtype=dtype)*.1+1
            for scale in [0,1,50]:
                values=x*scale
                expected=rms_reference(values,weight)
                actual=rms_norm(values,weight)
                torch.testing.assert_close(actual,expected,rtol=tol,atol=tol)
                error=float((actual.float()-expected.float()).abs().max()) if actual.numel() else 0
                errors.append({'dtype':str(dtype),'shape':list(shape),'scale':scale,'max_error':error})
    checks['rms_multiple_dtypes_shapes_scales']=True
    for rows,width in [(1,17),(17,257),(2,1536),(0,31)]:
        x=torch.randn(rows,width,device='cuda')*100
        torch.testing.assert_close(softmax(x),torch.softmax(x,dim=-1),atol=2e-6,rtol=2e-5)
    checks['stable_softmax_odd_tail']=True
    x=torch.randn(4,17,device='cuda').T;weight=torch.ones(4,device='cuda')
    try:rms_norm(x,weight)
    except ValueError:checks['noncontiguous_rejected']=True
    else:raise AssertionError('noncontiguous accepted')
    with torch.enable_grad():
        values=torch.randn(1,1536,device='cuda',requires_grad=True)
        try:rms_norm(values,torch.ones(1536,device='cuda'))
        except RuntimeError:checks['unsupported_autograd_rejected']=True
        else:raise AssertionError('missing backward accepted')
    save('31_correctness.json',{'checks':checks,'cases':errors,'scope':'Real CUDA Triton execution; finite test inputs, no backward verification'})
    compiled=torch.compile(rms_reference,dynamic=False)
    results=[]
    for rows in [1,32,512,2048]:
        x=torch.randn(rows,1536,device='cuda',dtype=torch.float16)
        weight=torch.ones(1536,device='cuda',dtype=torch.float16)
        torch.testing.assert_close(compiled(x,weight),rms_reference(x,weight),atol=3e-3,rtol=3e-3)
        modes={
            'torch_eager':lambda:rms_reference(x,weight),
            'torch_compile':lambda:compiled(x,weight),
            'triton_4_warps':lambda:rms_norm(x,weight,num_warps=4),
            'triton_8_warps':lambda:rms_norm(x,weight,num_warps=8),
        }
        measured={name:bench(fn) for name,fn in modes.items()}
        results.append({'shape':[rows,1536],'dtype':'float16','measurements':measured})
        print(rows,{name:round(value['wall_median_ms'],5) for name,value in measured.items()},flush=True)
        save('31_microbenchmark.json',{'results':results,'scope':'Isolated RMSNorm on warmed/cached inputs, includes wrapper and allocation; not inference acceleration ratio'})


if __name__=='__main__':
    main()
