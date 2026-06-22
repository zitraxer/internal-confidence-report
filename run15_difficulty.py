import numpy as np, glob, os, json, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
cv=StratifiedKFold(5,shuffle=True,random_state=0)
pipe=lambda: make_pipeline(StandardScaler(),LogisticRegression(C=0.05,max_iter=3000))
lr=lambda: make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000))

files=sorted(glob.glob("conf12_*.npz"))
names=[os.path.basename(f)[7:-4] for f in files]
D={n:np.load(f) for n,f in zip(names,files)}
Y=np.stack([D[n]["y"] for n in names])           # (12, 185) correctness, aligned by item
n_models=len(names)

def best_probe(X,y):
    aucs=[roc_auc_score(y,cross_val_predict(pipe(),X[:,L,:],y,cv=cv,method="predict_proba")[:,1]) for L in range(X.shape[1])]
    bL=int(np.argmax(aucs)); return cross_val_predict(pipe(),X[:,bL,:],y,cv=cv,method="predict_proba")[:,1]
def auc_feats(F,y):
    return roc_auc_score(y,cross_val_predict(lr(),F,y,cv=cv,method="predict_proba")[:,1])
def boot_diff(y,base_pred,full_pred,B=2000):
    rng=np.random.default_rng(0);pos=np.where(y==1)[0];neg=np.where(y==0)[0];d=[]
    for _ in range(B):
        bi=np.concatenate([rng.choice(pos,len(pos),True),rng.choice(neg,len(neg),True)])
        d.append(roc_auc_score(y[bi],full_pred[bi])-roc_auc_score(y[bi],base_pred[bi]))
    return np.percentile(d,2.5),np.percentile(d,97.5)

print(f"{'model':16s}{'err':>4s}{'diff-only':>10s}{'+token':>8s}{'+probe':>8s}{'tokΔ[95%CI]':>20s}{'verbal-only':>12s}")
res=[]
for i,n in enumerate(names):
    d=D[n]; X,y,ct,cvb=d["X"],d["y"],d["conf_tok"],d["conf_verb"]
    if y.sum()<8 or (1-y).sum()<8:
        print(f"{n:16s}{int((y==0).sum()):>4d}   (too few errors — skipped)"); res.append(dict(model=n,skipped=True)); continue
    diff=(Y.sum(0)-y)/(n_models-1)              # leave-one-out item difficulty
    probe_p=best_probe(X,y)
    # CV predictions for each feature-set
    base=cross_val_predict(lr(),diff[:,None],y,cv=cv,method="predict_proba")[:,1]
    full_t=cross_val_predict(lr(),np.c_[diff,ct],y,cv=cv,method="predict_proba")[:,1]
    full_p=cross_val_predict(lr(),np.c_[diff,probe_p],y,cv=cv,method="predict_proba")[:,1]
    a_d=roc_auc_score(y,base); a_dt=roc_auc_score(y,full_t); a_dp=roc_auc_score(y,full_p)
    lo,hi=boot_diff(y,base,full_t)
    a_vd=auc_feats(np.c_[diff,cvb],y)           # difficulty + verbalized
    sig="yes" if lo>0 else "no"
    print(f"{n:16s}{int((y==0).sum()):>4d}{a_d:>10.3f}{a_dt:>8.3f}{a_dp:>8.3f}   +{a_dt-a_d:.3f}[{lo:+.3f},{hi:+.3f}]{a_vd:>12.3f}")
    res.append(dict(model=n,err=int((y==0).sum()),diff_only=a_d,diff_plus_token=a_dt,diff_plus_probe=a_dp,
                    token_increment=a_dt-a_d,token_inc_ci=[lo,hi],token_adds=sig,diff_plus_verbal=a_vd))
json.dump(res,open("difficulty_results.json","w"),indent=2,default=float)
ok=[r for r in res if not r.get("skipped")]
if ok:
    print(f"\nAcross {len(ok)} error-rich models:")
    print(f"  difficulty alone predicts a model's correctness at mean AUROC {np.mean([r['diff_only'] for r in ok]):.3f}")
    print(f"  adding the model's OWN token signal -> {np.mean([r['diff_plus_token'] for r in ok]):.3f} (mean increment +{np.mean([r['token_increment'] for r in ok]):.3f})")
    print(f"  models where token adds significantly beyond difficulty (CI>0): {sum(r['token_adds']=='yes' for r in ok)}/{len(ok)}")
    print(f"  difficulty + verbalized -> {np.mean([r['diff_plus_verbal'] for r in ok]):.3f}")
print("DONE")
