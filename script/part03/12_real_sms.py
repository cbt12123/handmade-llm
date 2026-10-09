"""Optional real-data experiment; downloads UCI SMS once and caches the original TSV."""
import urllib.request,zipfile,io
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from _common import ROOT,np,split_indices,report

def main():
    folder=ROOT/'data'/'part03'; folder.mkdir(parents=True,exist_ok=True); path=folder/'SMSSpamCollection'
    if not path.exists():
        url='https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip'
        with urllib.request.urlopen(url,timeout=60) as response: archive=zipfile.ZipFile(io.BytesIO(response.read()))
        member=next(n for n in archive.namelist() if n.endswith('SMSSpamCollection')); path.write_bytes(archive.read(member))
    (folder/'README.md').write_text('Source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection\nAlmeida and Hidalgo (2011). DOI: 10.24432/C5CC84. CC BY 4.0.\nCached original text; never silently replaced by toy data.\n',encoding='utf-8')
    rows=[line.split('\t',1) for line in path.read_text(encoding='utf-8-sig').splitlines() if '\t' in line]
    groups={}
    for label,text in rows:
        key=' '.join(text.lower().split()); groups.setdefault(key,set()).add(label)
    clean=[(text,next(iter(labels))) for text,labels in groups.items() if len(labels)==1]; x=np.array([text for text,_ in clean]); y=np.array([int(label=='spam') for _,label in clean]); train,valid,test=split_indices(y)
    experiments=[]; best=None; best_score=-1
    from sklearn.metrics import f1_score
    for c in [.5,2.]:
        model=make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=2),LogisticRegression(C=c,class_weight='balanced',max_iter=1000)); model.fit(x[train],y[train]); score=f1_score(y[valid],model.predict(x[valid]))
        experiments.append({'C':c,'valid_spam_f1':float(score)})
        if score>best_score: best_score=score; best=model
    import joblib
    from _common import OUT
    joblib.dump(best,OUT/'12_sms.joblib'); prediction=best.predict(x[test]); assert np.array_equal(prediction,joblib.load(OUT/'12_sms.joblib').predict(x[test]))
    report('12_sms_report.json',{'original_rows':len(rows),'unique_unambiguous_rows':len(clean),'removed_rows':len(rows)-len(clean),'experiments':experiments,'test':classification_report(y[test],prediction,target_names=['ham','spam'],output_dict=True,zero_division=0),'reload_equal':True})

if __name__=='__main__': main()
