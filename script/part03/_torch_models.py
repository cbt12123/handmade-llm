"""Shared model definitions, CPU training and independent best-model selection."""
import copy
import random
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from _common import np, SEED

def seed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(2)

class DigitsMLP(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Flatten(),nn.Linear(64,64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,10))
    def forward(self,x): return self.net(x)

class DigitsCNN(nn.Module):
    def __init__(self):
        super().__init__(); self.features=nn.Sequential(nn.Conv2d(1,8,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(8,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2)); self.head=nn.Linear(16*2*2,10)
    def forward(self,x): return self.head(self.features(x).flatten(1))

def evaluate(model,x,y):
    model.eval()
    with torch.no_grad():
        logits=model(x); loss=nn.functional.cross_entropy(logits,y).item(); predictions=logits.argmax(1)
    return loss,float((predictions==y).float().mean()),predictions

def train_classifier(model,x,y,train,valid,epochs=25):
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=1e-4)
    loader=DataLoader(TensorDataset(x[train],y[train]),batch_size=64,shuffle=True,num_workers=0)
    history={key:[] for key in ["train_loss","valid_loss","train_acc","valid_acc"]}; best_loss=float("inf"); best=None
    for epoch in range(epochs):
        model.train()
        for xb,yb in loader:
            optimizer.zero_grad(); loss=nn.functional.cross_entropy(model(xb),yb); loss.backward(); optimizer.step()
        tl,ta,_=evaluate(model,x[train],y[train]); vl,va,_=evaluate(model,x[valid],y[valid])
        for key,value in zip(history,[tl,vl,ta,va]): history[key].append(value)
        if vl<best_loss: best_loss=vl; best=copy.deepcopy(model.state_dict())
    last_checkpoint={"epoch":epochs,"model":copy.deepcopy(model.state_dict()),"optimizer":optimizer.state_dict(),"torch_rng":torch.get_rng_state()}
    model.load_state_dict(best)
    return history,last_checkpoint

class TinySequenceModel(nn.Module):
    """One-layer causal Transformer trained on fixed-length reversal."""
    def __init__(self,vocab=16,d_model=32,heads=4,max_len=40):
        super().__init__(); self.vocab=vocab; self.d_model=d_model; self.heads=heads; self.max_len=max_len
        self.embedding=nn.Embedding(vocab,d_model); self.position=nn.Embedding(max_len,d_model)
        self.norm1=nn.LayerNorm(d_model); self.attention=nn.MultiheadAttention(d_model,heads,batch_first=True,dropout=0.)
        self.norm2=nn.LayerNorm(d_model); self.ff=nn.Sequential(nn.Linear(d_model,64),nn.GELU(),nn.Linear(64,d_model)); self.output=nn.Linear(d_model,vocab)
    def forward(self,ids,return_weights=False):
        length=ids.shape[1]
        if length>self.max_len: raise ValueError("sequence exceeds max_len")
        x=self.embedding(ids)+self.position(torch.arange(length,device=ids.device))[None]
        h=self.norm1(x); mask=torch.ones(length,length,dtype=torch.bool,device=ids.device).triu(1)
        context,weights=self.attention(h,h,h,attn_mask=mask,need_weights=return_weights,average_attn_weights=False)
        x=x+context; x=x+self.ff(self.norm2(x)); logits=self.output(x)
        return (logits,weights) if return_weights else logits
