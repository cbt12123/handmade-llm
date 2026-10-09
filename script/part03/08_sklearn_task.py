"""Offline digits classifier with leak-free CV, validation selection and reload."""
import joblib
from sklearn.datasets import load_digits
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from _common import np, plt, OUT, SEED, split_indices, metrics, report

data=load_digits(); x,y=data.data,data.target
train,valid,test=split_indices(y)
baseline=DummyClassifier(strategy="most_frequent").fit(x[train],y[train])
experiments=[]; candidates=[]
for c in [.1,1.]:
    model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=2000,random_state=SEED))
    cv=cross_val_score(model,x[train],y[train],cv=StratifiedKFold(3,shuffle=True,random_state=SEED))
    model.fit(x[train],y[train])
    score=model.score(x[valid],y[valid])
    experiments.append({"C":c,"train_cv_mean":float(cv.mean()),"validation_accuracy":float(score)})
    candidates.append(model)
best=int(np.argmax([run["validation_accuracy"] for run in experiments])); model=candidates[best]
prediction=model.predict(x[test]); joblib.dump(model,OUT/"08-model.joblib")
loaded=joblib.load(OUT/"08-model.joblib")
assert np.array_equal(prediction,loaded.predict(x[test]))
np.savez(OUT/"08-splits.npz",train=train,valid=valid,test=test)
fig,axes=plt.subplots(2,5,figsize=(10,4))
for ax,index,pred in zip(axes.flat,test[:10],prediction[:10]):
    ax.imshow(data.images[index],cmap="gray",vmin=0,vmax=16)
    ax.set_title(f"true={y[index]}, pred={pred}"); ax.axis("off")
fig.tight_layout(); fig.savefig(OUT/"08-predictions.png",dpi=150); plt.close(fig)
report("08-report.json",{"dataset":"sklearn load_digits, 8x8, 1797 samples", "split_sizes":[len(train),len(valid),len(test)],"experiments":experiments,"selected_C":experiments[best]["C"],"baseline_test":metrics(y[test],baseline.predict(x[test])),"test":metrics(y[test],prediction),"reload_identical":True})
print("PASS: splits, CV pipeline, selection and reload")
