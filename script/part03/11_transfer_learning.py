"""Optional pretrained ResNet18 experiment; downloads official weights (~44.7 MB)."""
import torch
from torch import nn
from torchvision.models import resnet18,ResNet18_Weights
from PIL import Image
from sklearn.datasets import load_digits
from _common import OUT,np,split_indices,metrics,report
from _torch_models import seed

def main():
    seed(); data=load_digits(); train,valid,test=split_indices(data.target); weights=ResNet18_Weights.DEFAULT; backbone=resnet18(weights=weights); backbone.fc=nn.Identity(); backbone.eval()
    transform=weights.transforms(); features=[]
    with torch.no_grad():
        for start in range(0,len(data.target),32):
            images=[transform(Image.fromarray((p/16*255).astype('uint8')).convert('RGB')) for p in data.images[start:start+32]]
            features.append(backbone(torch.stack(images)))
    features=torch.cat(features); labels=torch.tensor(data.target,dtype=torch.long); head=nn.Linear(512,10); optimizer=torch.optim.AdamW(head.parameters(),lr=.01)
    import copy
    best=float('inf'); state=None
    for _ in range(120):
        optimizer.zero_grad(); loss=nn.functional.cross_entropy(head(features[train]),labels[train]); loss.backward(); optimizer.step()
        with torch.no_grad(): value=nn.functional.cross_entropy(head(features[valid]),labels[valid]).item()
        if value<best: best=value; state=copy.deepcopy(head.state_dict())
    head.load_state_dict(state)
    with torch.no_grad(): prediction=head(features[test]).argmax(1).numpy()
    torch.save(state,OUT/'11_transfer_head.pt'); report('11_transfer_report.json',{'backbone':'ResNet18 IMAGENET1K_V1 frozen; 8x8 digits resized to 224','test':metrics(data.target[test],prediction),'note':'Transfer learning demonstration, not a claim of superiority to the small CNN.'})

if __name__=='__main__': main()
