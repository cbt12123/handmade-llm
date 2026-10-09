"""Causal next-token model on reversal; evaluate free generation, not just teacher forcing."""
import copy
import torch
from _common import OUT,np,plt,report
from _torch_models import seed,TinySequenceModel

def batch(generator,n,length=5):
    source=torch.randint(1,11,(n,length),generator=generator)
    full=torch.cat([source,torch.full((n,1),11),source.flip(1),torch.full((n,1),12)],1)
    inputs=full[:,:-1]; labels=full[:,1:].clone(); labels[:,:length]=-100
    return inputs,labels

def generate(model,source):
    ids=torch.cat([source,torch.full((len(source),1),11)],1)
    with torch.no_grad():
        for _ in range(source.shape[1]+1): ids=torch.cat([ids,model(ids)[:,-1].argmax(-1,keepdim=True)],1)
    return ids[:,source.shape[1]+1:]

def main():
    seed(); model=TinySequenceModel(); optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
    train_rng=torch.Generator().manual_seed(100); valid_rng=torch.Generator().manual_seed(200)
    vi,vy=batch(valid_rng,256); history=[]; best=float('inf'); state=None
    for step in range(1200):
        model.train(); x,y=batch(train_rng,64); optimizer.zero_grad(); loss=torch.nn.functional.cross_entropy(model(x).reshape(-1,16),y.reshape(-1),ignore_index=-100); loss.backward(); optimizer.step()
        if (step+1)%50==0:
            model.eval()
            with torch.no_grad(): value=torch.nn.functional.cross_entropy(model(vi).reshape(-1,16),vy.reshape(-1),ignore_index=-100).item()
            history.append([step+1,loss.item(),value])
            if value<best: best=value; state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state); model.eval(); torch.save({'model':state,'config':{'vocab':16,'d_model':32,'heads':4,'max_len':40}},OUT/'13_transformer.pt')
    results={}; examples=[]
    for length in [5,3,7]:
        generator=torch.Generator().manual_seed(300+length); source=torch.randint(1,11,(128,length),generator=generator); expected=torch.cat([source.flip(1),torch.full((128,1),12)],1); prediction=generate(model,source)
        results[str(length)]={'exact_sequence_accuracy':float((prediction==expected).all(1).float().mean()),'token_accuracy':float((prediction==expected).float().mean())}
        if length==5: examples=[{'source':source[i].tolist(),'expected':expected[i].tolist(),'generated':prediction[i].tolist()} for i in range(4)]
    saved=torch.load(OUT/'13_transformer.pt',weights_only=True); restored=TinySequenceModel(**saved['config']); restored.load_state_dict(saved['model']); restored.eval()
    probe=vi[:2,:6]
    with torch.no_grad():
        assert torch.allclose(model(vi[:2])[:,:6],model(probe),atol=1e-5), 'future leakage'
        assert torch.equal(generate(model,probe[:,:5]),generate(restored,probe[:,:5]))
        _,attention=model(vi[:1],return_weights=True)
        assert attention.triu(1).abs().max().item()<1e-7
    fig,axes=plt.subplots(1,2,figsize=(10,4)); h=np.array(history); axes[0].plot(h[:,0],h[:,1],label='train'); axes[0].plot(h[:,0],h[:,2],label='valid'); axes[0].legend(); axes[0].set(xlabel='Step',ylabel='Cross entropy')
    axes[1].imshow(attention[0,0].numpy(),vmin=0,vmax=1,cmap='Blues'); axes[1].set(title='Head 0 attention',xlabel='Key position',ylabel='Query position'); fig.tight_layout(); fig.savefig(OUT/'13_training_attention.png',dpi=150); plt.close(fig)
    report('13_report.json',{'task':'reverse symbols 1..10, SEP=11 EOS=12 PAD=0','train_length':5,'steps':1200,'generation':results,'examples':examples,'causal_prefix_check':True,'reload_equal':True,'best_valid_ce':best})

if __name__=='__main__': main()
