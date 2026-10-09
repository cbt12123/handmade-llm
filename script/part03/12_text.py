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
    train=[('这部电影很好看',1),('剧情精彩值得推荐',1),('演员表现很好',1),('我很喜欢这个故事',1),('画面漂亮非常喜欢',1),('故事精彩令人开心',1),('这部电影很难看',0),('剧情无聊不推荐',0),('演员表现很差',0),('我很讨厌这个故事',0),('画面糟糕非常失望',0),('故事无聊令人失望',0)]
    valid=[('电影精彩我很喜欢',1),('剧情糟糕我很失望',0)]
    test=[('故事很好值得推荐',1),('演员精彩令人开心',1),('电影无聊不推荐',0),('故事很差非常失望',0)]
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
    report('12_report.json',{'dataset':'explicit tiny synthetic Chinese teaching set','split_sizes':[len(train),len(valid),len(test)],'tfidf_test':metrics([y for _,y in test],baseline.predict(vectorizer.transform([s for s,_ in test]))),'gru_test':metrics([y for _,y in test],prediction),'generated_bigram':generated,'reload_equal':True,'vocab_size':len(vocab)})

if __name__=='__main__': main()
