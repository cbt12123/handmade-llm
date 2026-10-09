"""Digits CNN; optional --image path predicts a white digit on a black background."""
import argparse
import torch
from PIL import Image
from sklearn.datasets import load_digits
from _common import OUT, np, plt, split_indices, metrics, report, curves
from _torch_models import seed, DigitsCNN, train_classifier, evaluate

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--image'); args=parser.parse_args()
    if args.image:
        model=DigitsCNN(); model.load_state_dict(torch.load(OUT/'11_best.pt',weights_only=True)); model.eval()
        pixels=np.asarray(Image.open(args.image).convert('L').resize((8,8)),dtype=np.float32)/255
        with torch.no_grad(): print('prediction:',model(torch.tensor(pixels)[None,None]).argmax(1).item())
        return
    seed(); data=load_digits(); x=torch.tensor(data.images[:,None]/16,dtype=torch.float32); y=torch.tensor(data.target,dtype=torch.long)
    train,valid,test=split_indices(data.target); model=DigitsCNN(); history,_=train_classifier(model,x,y,train,valid,epochs=40)
    torch.save(model.state_dict(),OUT/'11_best.pt'); _,_,pred=evaluate(model,x[test],y[test])
    restored=DigitsCNN(); restored.load_state_dict(torch.load(OUT/'11_best.pt',weights_only=True)); assert torch.equal(pred,evaluate(restored,x[test],y[test])[2])
    with torch.no_grad(): maps=model.features[:2](x[test[:1]])[0].numpy()
    fig,axes=plt.subplots(2,4,figsize=(8,4))
    for ax,feature in zip(axes.flat,maps): ax.imshow(feature,cmap='viridis'); ax.axis('off')
    fig.tight_layout(); fig.savefig(OUT/'11_feature_maps.png',dpi=150); plt.close(fig)
    Image.fromarray((data.images[test[0]]/16*255).astype('uint8')).resize((160,160)).save(OUT/'11_example_digit.png')
    curves(history,'11_curves.png'); report('11_report.json',{'test':metrics(data.target[test],pred.numpy()),'reload_equal':True,'input_shape':list(x.shape),'errors':int((pred.numpy()!=data.target[test]).sum())})

if __name__=='__main__': main()
