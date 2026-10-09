"""Offline toy text classifier and character bigram generator."""
import torch
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from collections import Counter,defaultdict
from _common import OUT,np,report,metrics
from _torch_models import seed

class TextRNN(nn.Module):
    def __init__(self,vocab):
        super().__init__(); self.embedding=nn.Embedding(vocab,16,padding_idx=0); self.rnn=nn.GRU(16,24,batch_first=True); self.head=nn.Linear(24,2)
    def forward(self,ids,lengths):
        embedded=self.embedding(ids); packed=nn.utils.rnn.pack_padded_sequence(embedded,lengths.cpu(),batch_first=True,enforce_sorted=False)
        _,hidden=self.rnn(packed); return self.head(hidden[-1])

def main():
    seed()
    # Explicit splits: a small teaching set, not a real-world performance estimate.
    train=[('这道题我已经理解',1),('公式清楚我会计算',1),('步骤明白可以独立完成',1),('我已经学会这个方法',1),('例子清楚我已经懂了',1),('推导明白可以继续学习',1),('这道题我还没有理解',0),('公式不懂需要重新学习',0),('步骤不明白需要帮助',0),('我还不会这个方法',0),('例子看不懂需要复习',0),('推导不清楚我很困惑',0)]
    valid=[('这道题已经明白可以完成',1),('这道题不懂需要复习',0)]
    test=[('公式明白我已经学会',1),('方法清楚可以独立计算',1),('公式不明白我还不会',0),('方法不懂需要帮助',0)]
    text=[s for s,_ in train]; labels=torch.tensor([y for _,y in train])
    vocab={'<PAD>':0,'<UNK>':1}
    for char in sorted(set(''.join(text))): vocab[char]=len(vocab)
    def encode(items):
        sequences=[[vocab.get(c,1) for c in s] for s,_ in items]; lengths=torch.tensor([len(s) for s in sequences]); ids=torch.zeros(len(items),int(lengths.max()),dtype=torch.long)
        for i,s in enumerate(sequences): ids[i,:len(s)]=torch.tensor(s)
        return ids,lengths
    vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(1,2)); baseline=LogisticRegression().fit(vectorizer.fit_transform(text),labels.numpy())
    model=TextRNN(len(vocab)); optimizer=torch.optim.Adam(model.parameters(),lr=.01); ids,lengths=encode(train); vi,vl=encode(valid); best=float('inf'); state=None
    import copy
    for _ in range(180):
        model.train(); optimizer.zero_grad(); loss=nn.functional.cross_entropy(model(ids,lengths),labels); loss.backward(); optimizer.step(); model.eval()
        with torch.no_grad(): value=nn.functional.cross_entropy(model(vi,vl),torch.tensor([y for _,y in valid])).item()
        if value<best: best=value; state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state); model.eval(); ti,tl=encode(test)
    with torch.no_grad(): prediction=model(ti,tl).argmax(1).numpy()
    torch.save({'model':state,'vocab':vocab},OUT/'12_text.pt')
    restored=TextRNN(len(vocab)); restored.load_state_dict(torch.load(OUT/'12_text.pt',weights_only=True)['model']); restored.eval()
    with torch.no_grad(): assert np.array_equal(prediction,restored(ti,tl).argmax(1).numpy())
    counts=defaultdict(Counter)
    for sentence in text:
        for a,b in zip('^'+sentence,sentence+'$'): counts[a][b]+=1
    rng=np.random.default_rng(42); current='^'; generated=''
    for _ in range(30):
        choices=list(counts[current]); probability=np.array([counts[current][c] for c in choices],float); probability/=probability.sum(); current=rng.choice(choices,p=probability)
        if current=='$': break
        generated+=current
    report('12_report.json',{'dataset':'explicit tiny synthetic Chinese learning-feedback set','label_meaning':{'0':'needs review','1':'self-reported understood'},'split_sizes':[len(train),len(valid),len(test)],'tfidf_test':metrics([y for _,y in test],baseline.predict(vectorizer.transform([s for s,_ in test]))),'gru_test':metrics([y for _,y in test],prediction),'test_examples':[{'text':s,'label':y,'prediction':int(p)} for (s,y),p in zip(test,prediction)],'generated_bigram':generated,'reload_equal':True,'vocab_size':len(vocab)})

if __name__=='__main__': main()
