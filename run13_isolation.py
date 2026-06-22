import numpy as np, glob, os, json, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
pipe=lambda: make_pipeline(StandardScaler(),LogisticRegression(C=0.05,max_iter=2000))
NCAP=85  # first 85 items are capitals, rest atomic
caps=["France","Japan","Brazil","Egypt","Canada","Australia","Turkey","Kazakhstan","Nigeria","Myanmar","Bhutan","Nauru","Kiribati","Tuvalu","Palau","Vanuatu","Eritrea","Djibouti","Comoros","Lesotho","Eswatini","Suriname","Guyana","Belize","Brunei","Mongolia","Laos","Cambodia","Nepal","SriLanka","Kyrgyzstan","Tajikistan","Turkmenistan","Azerbaijan","Armenia","Georgia","Moldova","Slovenia","Slovakia","Croatia","Serbia","Albania","Macedonia","Montenegro","Latvia","Lithuania","Estonia","Iceland","Malta","Cyprus","Senegal","Mali","Niger","Chad","Burkina Faso","Benin","Togo","Gabon","Cameroon","Angola","Zambia","Malawi","Mozambique","Madagascar","Botswana","Namibia","Rwanda","Burundi","Uganda","Tanzania","Paraguay","Uruguay","Bolivia","Ecuador","Honduras","Nicaragua","Panama","Jamaica","Bahamas","Qatar","Oman","Yemen","Jordan","Lebanon","Bahrain"]
els="Hydrogen Helium Lithium Beryllium Boron Carbon Nitrogen Oxygen Fluorine Neon Sodium Magnesium Aluminium Silicon Phosphorus Sulfur Chlorine Argon Potassium Calcium Scandium Titanium Vanadium Chromium Manganese Iron Cobalt Nickel Copper Zinc Gallium Germanium Arsenic Selenium Bromine Krypton Rubidium Strontium Yttrium Zirconium Niobium Molybdenum Technetium Ruthenium Rhodium Palladium Silver Cadmium Indium Tin Antimony Tellurium Iodine Xenon Caesium Barium Lanthanum Cerium Praseodymium Neodymium Promethium Samarium Europium Gadolinium Terbium Dysprosium Holmium Erbium Thulium Ytterbium Lutetium Hafnium Tantalum Tungsten Rhenium Osmium Iridium Platinum Gold Mercury Thallium Lead Bismuth Polonium Astatine Radon Francium Radium Actinium Thorium Protactinium Uranium Neptunium Plutonium Americium Curium Berkelium Californium Einsteinium Fermium".split()
txt=[f"capital city of {c}" for c in caps]+[f"atomic number of {e}" for e in els]

def both(y): return y.sum()>=4 and (1-y).sum()>=4
def probe_transfer(Xs,ys,Xt,yt):
    best=-1;bL=Xs.shape[1]//2
    for L in range(Xs.shape[1]):
        try:
            a=roc_auc_score(ys,cross_val_predict(pipe(),Xs[:,L,:],ys,cv=3,method="predict_proba")[:,1])
        except Exception: a=0.5
        if a>best: best=a;bL=L
    clf=pipe().fit(Xs[:,bL,:],ys); return roc_auc_score(yt,clf.predict_proba(Xt[:,bL,:])[:,1])
def text_transfer(ts,ys,tt,yt):
    v=TfidfVectorizer(analyzer="char_wb",ngram_range=(2,4)).fit(ts)
    clf=LogisticRegression(C=1.0,max_iter=2000).fit(v.transform(ts),ys)
    return roc_auc_score(yt,clf.predict_proba(v.transform(tt))[:,1])
def boot(y,s,B=2000):
    rng=np.random.default_rng(0);pos=np.where(y==1)[0];neg=np.where(y==0)[0];o=[]
    for _ in range(B):
        bi=np.concatenate([rng.choice(pos,len(pos),True),rng.choice(neg,len(neg),True)]);o.append(roc_auc_score(y[bi],s[bi]))
    return np.percentile(o,2.5),np.percentile(o,97.5)

print(f"{'model':16s}{'capErr':>7s}{'atomErr':>8s}{'PROBE cap->atom':>16s}{'PROBE atom->cap':>16s}{'TEXT cross':>11s}")
res=[]
for f in sorted(glob.glob("conf12_*.npz")):
    s=os.path.basename(f)[7:-4]; d=np.load(f); X,y=d["X"],d["y"]
    Xc,yc,Xa,ya=X[:NCAP],y[:NCAP],X[NCAP:],y[NCAP:]
    tc,ta=txt[:NCAP],txt[NCAP:]
    if not(both(yc) and both(ya)):
        print(f"{s:16s}{int((yc==0).sum()):>7d}{int((ya==0).sum()):>8d}   (too few errors in a domain — skipped)")
        res.append(dict(model=s,skipped=True,capErr=int((yc==0).sum()),atomErr=int((ya==0).sum()))); continue
    pca=probe_transfer(Xc,yc,Xa,ya); pac=probe_transfer(Xa,ya,Xc,yc)
    txc=0.5*(text_transfer(tc,yc,ta,ya)+text_transfer(ta,ya,tc,yc))
    print(f"{s:16s}{int((yc==0).sum()):>7d}{int((ya==0).sum()):>8d}{pca:>16.3f}{pac:>16.3f}{txc:>11.3f}")
    res.append(dict(model=s,capErr=int((yc==0).sum()),atomErr=int((ya==0).sum()),
                    probe_cap2atom=pca,probe_atom2cap=pac,text_cross=txc,skipped=False))
json.dump(res,open("isolation_results.json","w"),indent=2,default=float)
ok=[r for r in res if not r.get("skipped")]
if ok:
    pm=np.mean([0.5*(r["probe_cap2atom"]+r["probe_atom2cap"]) for r in ok])
    tm=np.mean([r["text_cross"] for r in ok])
    print(f"\nAcross {len(ok)} error-rich models -- mean cross-domain PROBE AUROC = {pm:.3f} | mean cross-domain TEXT baseline = {tm:.3f}")
    print("If PROBE stays well above 0.5 while TEXT collapses to ~0.5, the probe reads a domain-general internal signal, not entity/topic lookup.")
