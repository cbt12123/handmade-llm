"""CPU MLP: gradients, best inference weights and resumable last checkpoint."""
import torch
from sklearn.datasets import load_digits
from _common import OUT, split_indices, metrics, report, curves, np
from _torch_models import seed, DigitsMLP, train_classifier, evaluate

def main():
    seed(); data=load_digits(); x=torch.tensor(data.data/16,dtype=torch.float32); y=torch.tensor(data.target,dtype=torch.long)
    train,valid,test=split_indices(data.target)
    a=torch.tensor([2.,3.],requires_grad=True); (a.square().sum()).backward()
    assert torch.equal(a.grad,torch.tensor([4.,6.]))
    # Compare the chapter-9 manual chain rule against torch autograd, float64.
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('numpy_network',Path(__file__).with_name('09_numpy_network.py')); manual=importlib.util.module_from_spec(spec); spec.loader.exec_module(manual)
    rng=np.random.default_rng(42); sample=rng.normal(size=(5,2)); target=np.array([0.,1.,0.,1.,1.])
    parameters={'W1':rng.normal(size=(2,3))*.1,'b1':np.zeros(3),'W2':rng.normal(size=(3,1))*.1,'b2':np.zeros(1)}
    _,gradient=manual.loss_grad(sample,target,parameters); tensors={k:torch.tensor(v,requires_grad=True) for k,v in parameters.items()}
    hidden=torch.tanh(torch.tensor(sample)@tensors['W1']+tensors['b1']); z=(hidden@tensors['W2']+tensors['b2']).ravel()
    torch.nn.functional.binary_cross_entropy_with_logits(z,torch.tensor(target)).backward()
    parity=max(float(np.max(np.abs(gradient[k]-v.grad.numpy()))) for k,v in tensors.items()); assert parity<1e-10
    model=DigitsMLP(); history,last=train_classifier(model,x,y,train,valid)
    torch.save(model.state_dict(),OUT/'10_best.pt'); torch.save(last,OUT/'10_resume.pt')
    _,_,pred=evaluate(model,x[test],y[test]); restored=DigitsMLP(); restored.load_state_dict(torch.load(OUT/'10_best.pt',weights_only=True))
    assert torch.equal(pred,evaluate(restored,x[test],y[test])[2])
    # Restore the last training state; this differs from the best inference state.
    resumed=DigitsMLP(); checkpoint=torch.load(OUT/'10_resume.pt',weights_only=True)
    resumed.load_state_dict(checkpoint['model']); optimizer=torch.optim.AdamW(resumed.parameters(),lr=.003,weight_decay=1e-4)
    optimizer.load_state_dict(checkpoint['optimizer']); torch.set_rng_state(checkpoint['torch_rng'])
    resumed.train(); optimizer.zero_grad(); loss=torch.nn.functional.cross_entropy(resumed(x[train[:64]]),y[train[:64]])
    loss.backward(); optimizer.step(); assert torch.isfinite(loss)
    curves(history,'10_curves.png'); report('10_report.json',{'torch':torch.__version__,'numpy_autograd_max_error':parity,'test':metrics(data.target[test],pred.numpy()),'reload_equal':True,'resume_one_step_loss':loss.item(),'epochs':checkpoint['epoch']})

if __name__=='__main__': main()
