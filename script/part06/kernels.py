"""Original forward-only Triton kernels: masked indexing, reduction, fusion."""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _add(A,B,C,N:tl.constexpr,BLOCK:tl.constexpr):
    offsets=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    mask=offsets<N
    a=tl.load(A+offsets,mask=mask,other=0)
    b=tl.load(B+offsets,mask=mask,other=0)
    tl.store(C+offsets,a+b,mask=mask)


def vector_add(a,b):
    if a.device != b.device or not a.is_cuda or a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError('需要相同形状、类型与GPU的两个输入')
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError('演示仅支持连续输入')
    if torch.is_grad_enabled() and (a.requires_grad or b.requires_grad):
        raise RuntimeError('本算子仅用于前向推理')
    out=torch.empty_like(a)
    if a.numel():
        with torch.cuda.device(a.device):
            _add[(triton.cdiv(a.numel(),256),)](a,b,out,a.numel(),BLOCK=256)
    return out


@triton.jit
def _rms(X,W,Y,D:tl.constexpr,EPS:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0)
    cols=tl.arange(0,BLOCK)
    values=tl.load(X+row*D+cols,mask=cols<D,other=0).to(tl.float32)
    inverse=tl.rsqrt(tl.sum(values*values,axis=0)/D+EPS)
    # Qwen casts normalized values back BEFORE multiplying its weight.
    normalized=(values*inverse).to(Y.dtype.element_ty)
    weight=tl.load(W+cols,mask=cols<D,other=0)
    result=normalized.to(tl.float32)*weight.to(tl.float32)
    tl.store(Y+row*D+cols,result,mask=cols<D)


def rms_supported(x,weight):
    return (x.is_cuda and weight.is_cuda and x.device==weight.device and x.ndim>=1
            and x.is_contiguous() and weight.is_contiguous() and weight.ndim==1
            and x.shape[-1]==weight.numel() and 1<=x.shape[-1]<=8192
            and x.dtype in (torch.float16,torch.bfloat16,torch.float32) and weight.dtype==x.dtype)


def rms_norm(x,weight,eps=1e-6,num_warps=None):
    if not rms_supported(x,weight):
        raise ValueError('RMSNorm 需要连续CUDA张量，匹配一维权重，支持维度1～8192与FP16/BF16/FP32')
    if not math.isfinite(eps) or eps<=0:
        raise ValueError('eps 必须为有限正数')
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        raise RuntimeError('本算子仅用于推理，没有注册反向传播')
    width=x.shape[-1]
    output=torch.empty_like(x)
    rows=x.numel()//width
    if rows:
        block=triton.next_power_of_2(width)
        warps=num_warps or (4 if block<=4096 else 8)
        if warps not in (4,8):
            raise ValueError('本实验只开放 num_warps=4 或 8')
        with torch.cuda.device(x.device):
            _rms[(rows,)](x,weight,output,width,eps,BLOCK=block,num_warps=warps,enable_fp_fusion=False)
    return output


@triton.jit
def _softmax(X,Y,D:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);cols=tl.arange(0,BLOCK)
    x=tl.load(X+row*D+cols,mask=cols<D,other=-float('inf')).to(tl.float32)
    numerator=tl.exp(x-tl.max(x,axis=0))
    y=numerator/tl.sum(numerator,axis=0)
    tl.store(Y+row*D+cols,y,mask=cols<D)


def softmax(x):
    if not x.is_cuda or x.ndim!=2 or not x.is_contiguous() or not 1<=x.shape[-1]<=8192:
        raise ValueError('softmax 教学实现只接受连续二维CUDA输入，列数1～8192')
    if x.dtype not in (torch.float16,torch.bfloat16,torch.float32):
        raise ValueError('只支持浮点输入')
    if torch.is_grad_enabled() and x.requires_grad:
        raise RuntimeError('仅用于推理')
    out=torch.empty_like(x)
    if x.shape[0]:
        with torch.cuda.device(x.device):
            _softmax[(x.shape[0],)](x,out,x.shape[-1],BLOCK=triton.next_power_of_2(x.shape[-1]),num_warps=4)
    return out
