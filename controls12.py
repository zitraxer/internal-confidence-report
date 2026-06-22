import numpy as np, glob, os, json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
cv=StratifiedKFold(5,shuffle=True,random_state=0)
pipe=lambda: make_pipeline(StandardScaler(),LogisticRegression(C=0.05,max_iter=3000))

# reconstruct text + kind in the EXACT order run12.py used (85 capitals then 100 elements)
capitals=["France","Japan","Brazil","Egypt","Canada","Australia","Turkey","Kazakhstan","Nigeria","Myanmar","Bhutan","Nauru","Kiribati","Tuvalu","Palau","Vanuatu","Eritrea","Djibouti","Comoros","Lesotho","Eswatini","Suriname","Guyana","Belize","Brunei","Mongolia","Laos","Cambodia","Nepal","SriLanka","Kyrgyzstan","Tajikistan","Turkmenistan","Azerbaijan","Armenia","Georgia","Moldova","Slovenia","Slovakia","Croatia","Serbia","Albania","Macedonia","Montenegro","Latvia","Lithuania","Estonia","Iceland","Malta","Cyprus","Senegal","Mali","Niger","Chad","Burkina Faso","Benin","Togo","Gabon","Cameroon","Angola","Zambia","Malawi","Mozambique","Madagascar","Botswana","Namibia","Rwanda","Burundi","Uganda","Tanzania","Paraguay","Uruguay","Bolivia","Ecuador","Honduras","Nicaragua","Panama","Jamaica","Bahamas","Qatar","Oman","Yemen","Jordan","Lebanon","Bahrain"]
elements="Hydrogen Helium Lithium Beryllium Boron Carbon Nitrogen Oxygen Fluorine Neon Sodium Magnesium Aluminium Silicon Phosphorus Sulfur Chlorine Argon Potassium Calcium Scandium Titanium Vanadium Chromium Manganese Iron Cobalt Nickel Copper Zinc Gallium Germanium Arsenic Selenium Bromine Krypton Rubidium Strontium Yttrium Zirconium Niobium Molybdenum Technetium Ruthenium Rhodium Palladium Silver Cadmium Indium Tin Antimony Tellurium Iodine Xenon Caesium Barium Lanthanum Cerium Praseodymium Neodymium Promethium Samarium Europium Gadolinium Terbium Dysprosium Holmium Erbium Thulium Ytterbium Lutetium Hafnium Tantalum Tungsten Rhenium Osmium Iridium Platinum Gold Mercury Thallium Lead Bismuth Polonium Astatine Radon Francium Radium Actinium Thorium Protactinium Uranium Neptunium Plutonium Americium Curium Berkelium Californium Einsteinium Fermium".split()
text=[f"capital city of {c}" for c in capitals]+[f"atomic number of {e}" for e in elements]
kind=np.array(["capital"]*len(capitals)+["atomic"]*len(elements))

def boot(y,s,B=2000):
    rng=np.random.default_rng(0); pos=np.where(y==1)[0]; neg=np.where(y==0)[0]; out=[]
    for _ in range(B):
        bi=np.concatenate([rng.choice(pos,len(pos),True),rng.choice(neg,len(neg),True)]); out.append(roc_auc_score(y[bi],s[bi]))
    return np.percentile(out,2.5),np.percentile(out,97.5)
def bestprobe(X,y):
    aucs=[roc_auc_score(y,cross_val_predict(pipe(),X[:,L,:],y,cv=cv,method="predict_proba")[:,1]) for L in range(X.shape[1])]
    bL=int(np.argmax(aucs)); return cross_val_predict(pipe(),X[:,bL,:],y,cv=cv,method="predict_proba")[:,1], max(aucs)

print(f"{'model':16s}{'N':>4s}{'err':>4s}{'verbal[95%CI]':>20s}{'token':>7s}{'probe':>7s}{'TEXT':>6s}{'shuf':>6s}{'atom-prb':>9s}")
res=[]
for f in sorted(glob.glob("conf12_*.npz")):
    s=os.path.basename(f)[7:-4]; d=np.load(f); X,y,ct,cvb=d["X"],d["y"],d["conf_tok"],d["conf_verb"]
    probe_p,ap=bestprobe(X,y)
    vlo,vhi=boot(y,cvb)
    # control: surface text only
    txt=cross_val_predict(make_pipeline(TfidfVectorizer(analyzer="char_wb",ngram_range=(2,4)),LogisticRegression(C=1.0,max_iter=2000)),text,y,cv=cv,method="predict_proba")[:,1]
    text_auc=roc_auc_score(y,txt)
    # control: shuffled labels for probe
    ysh=np.random.default_rng(0).permutation(y)
    shuf=roc_auc_score(ysh,cross_val_predict(pipe(),X[:,X.shape[1]//2,:],ysh,cv=cv,method="predict_proba")[:,1])
    # within-family (atomic only) probe
    msk=kind=="atomic"; 
    try: _,atom=bestprobe(X[msk],y[msk]); atom=f"{atom:.3f}"
    except Exception: atom="n/a"
    av,at,apv=roc_auc_score(y,cvb),roc_auc_score(y,ct),ap
    print(f"{s:16s}{len(y):>4d}{int((y==0).sum()):>4d}   {av:.3f}[{vlo:.2f},{vhi:.2f}]{at:>7.3f}{apv:>7.3f}{text_auc:>6.2f}{shuf:>6.2f}{atom:>9s}")
    res.append(dict(model=s,n=int(len(y)),err=int((y==0).sum()),verbal=av,verbal_ci=[vlo,vhi],token=at,probe=apv,text_baseline=text_auc,shuffled=shuf,atomic_probe=atom))
json.dump(res,open("controls12_results.json","w"),indent=2,default=float)
print("\nKEY: TEXT (surface-text baseline) and shuf (shuffled labels) should both be ~0.50 if we're isolating the model's STATE, not artifacts.")
