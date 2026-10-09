"""Two-layer NumPy MLP: gradient check, independent evaluation, reload."""
from sklearn.datasets import make_moons
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from _common import np, plt, OUT, SEED, split_indices, metrics, report, curves

def forward(x,p):
    hidden=np.tanh(x@p["W1"]+p["b1"])
    logits=(hidden@p["W2"]+p["b2"]).ravel()
    probability=1/(1+np.exp(-np.clip(logits,-50,50)))
    return hidden,logits,probability

def loss_grad(x,y,p):
    hidden,logits,prob=forward(x,p)
    loss=float(np.mean(np.logaddexp(0,logits)-y*logits))
    dz=((prob-y)/len(y))[:,None]
    dh=(dz@p["W2"].T)*(1-hidden**2)
    return loss,{"W2":hidden.T@dz,"b2":dz.sum(axis=0),"W1":x.T@dh,"b1":dh.sum(axis=0)}

def digits_experiment():
    """Main story: distinguish handwritten 3 (label 0) from 8 (label 1)."""
    data = load_digits()
    selected = np.isin(data.target, [3, 8])
    raw = data.data[selected].astype(np.float64)
    labels = (data.target[selected] == 8).astype(np.int64)
    train, valid, test = split_indices(labels)
    scaler = StandardScaler().fit(raw[train])
    x = scaler.transform(raw)
    rng = np.random.default_rng(SEED)
    parameters = {
        "W1": rng.normal(0, 0.12, (64, 16)), "b1": np.zeros(16),
        "W2": rng.normal(0, 0.12, (16, 1)), "b2": np.zeros(1),
    }
    best_loss = float("inf")
    best = None
    history = {key: [] for key in ["train_loss", "valid_loss", "train_acc", "valid_acc"]}
    for step in range(1000):
        _, gradients = loss_grad(x[train], labels[train], parameters)
        for name in parameters:
            parameters[name] -= 0.05 * gradients[name]
        if step % 10 == 0:
            train_loss = loss_grad(x[train], labels[train], parameters)[0]
            valid_loss = loss_grad(x[valid], labels[valid], parameters)[0]
            values = [train_loss, valid_loss]
            values += [float(np.mean((forward(x[indices], parameters)[2] >= .5) == labels[indices])) for indices in [train, valid]]
            for key, value in zip(history, values):
                history[key].append(value)
            if valid_loss < best_loss:
                best_loss = valid_loss
                best = {name: array.copy() for name, array in parameters.items()}
    prediction = forward(x[test], best)[2] >= .5
    np.savez(OUT / "09-digits-model.npz", **best, mean=scaler.mean_, scale=scaler.scale_)
    with np.load(OUT / "09-digits-model.npz") as saved:
        restored_x = (raw[test] - saved["mean"]) / saved["scale"]
        restored = {name: saved[name] for name in best}
        assert np.array_equal(prediction, forward(restored_x, restored)[2] >= .5)
    curves(history, "09-digits-training.png")
    return {"classes": {"0": 3, "1": 8}, "split_sizes": [len(train), len(valid), len(test)],
            "test": metrics(labels[test], prediction), "reload_equal": True,
            "best_validation_loss": best_loss}

def main():
    x,y=make_moons(n_samples=600,noise=.18,random_state=SEED)
    train,valid,test=split_indices(y); scaler=StandardScaler().fit(x[train]); x=scaler.transform(x)
    rng=np.random.default_rng(SEED)
    p={"W1":rng.normal(0,.3,(2,16)),"b1":np.zeros(16),"W2":rng.normal(0,.3,(16,1)),"b2":np.zeros(1)}
    _,analytic=loss_grad(x[train[:5]],y[train[:5]],p)
    errors=[]; eps=1e-5
    for name,array in p.items():
        for index in np.ndindex(array.shape):
            old=array[index]; array[index]=old+eps; plus=loss_grad(x[train[:5]],y[train[:5]],p)[0]
            array[index]=old-eps; minus=loss_grad(x[train[:5]],y[train[:5]],p)[0]; array[index]=old
            errors.append(abs((plus-minus)/(2*eps)-analytic[name][index]))
    assert max(errors)<1e-6
    history={key:[] for key in ["train_loss","valid_loss","train_acc","valid_acc"]}; best_loss=float("inf"); best=None
    for step in range(2000):
        _,grads=loss_grad(x[train],y[train],p)
        for name in p: p[name]-=.1*grads[name]
        if step%20==0:
            tl=loss_grad(x[train],y[train],p)[0]; vl=loss_grad(x[valid],y[valid],p)[0]
            history["train_loss"].append(tl); history["valid_loss"].append(vl)
            for indices,key in [(train,"train_acc"),(valid,"valid_acc")]: history[key].append(float(np.mean((forward(x[indices],p)[2]>=.5)==y[indices])))
            if vl<best_loss: best_loss=vl; best={name:array.copy() for name,array in p.items()}
    baseline=LogisticRegression().fit(x[train],y[train])
    prediction=forward(x[test],best)[2]>=.5
    np.savez(OUT/"09-model.npz",**best,mean=scaler.mean_,scale=scaler.scale_)
    with np.load(OUT/"09-model.npz") as saved:
        reloaded={name:saved[name] for name in best}
        raw_test=scaler.inverse_transform(x[test]); restored_input=(raw_test-saved['mean'])/saved['scale']
    assert np.array_equal(prediction,forward(restored_input,reloaded)[2]>=.5)
    curves(history,"09-training.png")
    xx,yy=np.meshgrid(np.linspace(-2.5,2.5,160),np.linspace(-2.5,2.5,160)); grid=np.column_stack([xx.ravel(),yy.ravel()]); proba=forward(grid,best)[2].reshape(xx.shape)
    fig,ax=plt.subplots(figsize=(6,5)); ax.contourf(xx,yy,proba,levels=20,cmap="RdBu",alpha=.6); ax.scatter(x[test,0],x[test,1],c=y[test],cmap="RdBu",s=15)
    ax.set(title="NumPy MLP: test data and decision probability",xlabel="Scaled feature 1",ylabel="Scaled feature 2"); fig.tight_layout(); fig.savefig(OUT/"09-boundary.png",dpi=150); plt.close(fig)
    digits_result = digits_experiment()
    report("09-report.json",{"gradient_max_error":float(max(errors)),"split_sizes":[len(train),len(valid),len(test)],"baseline_test":metrics(y[test],baseline.predict(x[test])),"mlp_test":metrics(y[test],prediction),"best_validation_loss":best_loss,"reload_identical":True,"digits_binary":digits_result,"moons_role":"auxiliary two-dimensional boundary illustration"})

if __name__=="__main__": main()
