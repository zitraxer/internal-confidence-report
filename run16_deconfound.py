import numpy as np, glob, os, json, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
cv=StratifiedKFold(5,shuffle=True,random_state=0)
probe_pipe=lambda: make_pipeline(StandardScaler(),LogisticRegression(C=0.05,max_iter=2000))
lr_pipe=lambda: make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000))

caps=["France","Japan","Brazil","Egypt","Canada","Australia","Turkey","Kazakhstan","Nigeria","Myanmar","Bhutan","Nauru","Kiribati","Tuvalu","Palau","Vanuatu","Eritrea","Djibouti","Comoros","Lesotho","Eswatini","Suriname","Guyana","Belize","Brunei","Mongolia","Laos","Cambodia","Nepal","SriLanka","Kyrgyzstan","Tajikistan","Turkmenistan","Azerbaijan","Armenia","Georgia","Moldova","Slovenia","Slovakia","Croatia","Serbia","Albania","Macedonia","Montenegro","Latvia","Lithuania","Estonia","Iceland","Malta","Cyprus","Senegal","Mali","Niger","Chad","Burkina Faso","Benin","Togo","Gabon","Cameroon","Angola","Zambia","Malawi","Mozambique","Madagascar","Botswana","Namibia","Rwanda","Burundi","Uganda","Tanzania","Paraguay","Uruguay","Bolivia","Ecuador","Honduras","Nicaragua","Panama","Jamaica","Bahamas","Qatar","Oman","Yemen","Jordan","Lebanon","Bahrain"]
els="Hydrogen Helium Lithium Beryllium Boron Carbon Nitrogen Oxygen Fluorine Neon Sodium Magnesium Aluminium Silicon Phosphorus Sulfur Chlorine Argon Potassium Calcium Scandium Titanium Vanadium Chromium Manganese Iron Cobalt Nickel Copper Zinc Gallium Germanium Arsenic Selenium Bromine Krypton Rubidium Strontium Yttrium Zirconium Niobium Molybdenum Technetium Ruthenium Rhodium Palladium Silver Cadmium Indium Tin Antimony Tellurium Iodine Xenon Caesium Barium Lanthanum Cerium Praseodymium Neodymium Promethium Samarium Europium Gadolinium Terbium Dysprosium Holmium Erbium Thulium Ytterbium Lutetium Hafnium Tantalum Tungsten Rhenium Osmium Iridium Platinum Gold Mercury Thallium Lead Bismuth Polonium Astatine Radon Francium Radium Actinium Thorium Protactinium Uranium Neptunium Plutonium Americium Curium Berkelium Californium Einsteinium Fermium".split()
NCAP=len(caps)
qlen=np.array([len(f"What is the capital city of {c}?") for c in caps]+[len(f"What is the atomic number of the element {e}?") for e in els],dtype=float)
typ=np.array([0]*NCAP+[1]*len(els),dtype=float)

files=sorted(glob.glob("conf12_*.npz")); names=[os.path.basename(f)[7:-4] for f in files]
Y=np.stack([np.load(f)["y"] for f in files])  # (12,185)
nM=len(names)

def auc_feats(F,y): return roc_auc_score(y,cross_val_predict(lr_pipe(),F,y,cv=cv,method="predict_proba")[:,1])
NLAY=12  # downsampled depths for the heatmap
conds=["raw (vs chance)","beyond difficulty","beyond format","beyond all confounds"]

grid={}; master=np.zeros((nM,4)); rawbest=np.zeros(nM); deconfbest=np.zeros(nM); errs=[]
for mi,(f,nm) in enumerate(zip(files,names)):
    d=np.load(f); X,y=d["X"],d["y"]; errs.append(int((y==0).sum()))
    diff=(Y.sum(0)-y)/(nM-1)
    Cdiff=diff[:,None]; Cfmt=np.c_[typ,qlen]; Call=np.c_[diff,typ,qlen]
    a_diff=auc_feats(Cdiff,y); a_fmt=auc_feats(Cfmt,y); a_all=auc_feats(Call,y)
    L=X.shape[1]; layers=np.unique(np.linspace(0,L-1,NLAY).astype(int))
    H=np.zeros((4,len(layers)))
    for j,ly in enumerate(layers):
        ps=cross_val_predict(probe_pipe(),X[:,ly,:],y,cv=cv,method="predict_proba")[:,1]
        H[0,j]=roc_auc_score(y,ps)-0.5
        H[1,j]=auc_feats(np.c_[diff,ps],y)-a_diff
        H[2,j]=auc_feats(np.c_[typ,qlen,ps],y)-a_fmt
        H[3,j]=auc_feats(np.c_[diff,typ,qlen,ps],y)-a_all
    grid[nm]=dict(H=H.tolist(),layers=layers.tolist(),L=int(L))
    bj=int(np.argmax(H[0]))             # best layer by raw signal
    master[mi]=H[:,bj]; rawbest[mi]=H[0,bj]; deconfbest[mi]=H[3,bj]
    print(f"[{nm}] err={errs[-1]:3d} raw+{H[0,bj]:.3f} | beyond-all +{H[3,bj]:.3f}",flush=True)

json.dump(dict(names=names,errs=errs,master=master.tolist(),grid=grid),open("deconfound_results.json","w"),indent=2,default=float)

order=np.argsort(deconfbest)[::-1]
vmax=max(0.3,float(master.max()))
# ---- master heatmap ----
fig,ax=plt.subplots(figsize=(8.5,7))
im=ax.imshow(master[order],aspect="auto",cmap="magma",vmin=0,vmax=vmax)
ax.set_xticks(range(4)); ax.set_xticklabels(conds,rotation=20,ha="right",fontsize=10)
ax.set_yticks(range(nM)); ax.set_yticklabels([f"{names[i]} (err {errs[i]})" for i in order],fontsize=9)
for i in range(nM):
    for j in range(4):
        v=master[order][i,j]; ax.text(j,i,f"{v:.2f}",ha="center",va="center",color="white" if v<vmax*0.6 else "black",fontsize=8)
ax.set_title("Probe signal that survives deconfounding (AUROC above baseline)\nbright = genuine signal beyond confounds; dark = it was format/difficulty")
fig.colorbar(im,label="signal (AUROC above baseline)"); plt.tight_layout(); plt.savefig("fig_deconf_master.png",dpi=130)

# ---- per-model grid of heatmaps ----
fig,axes=plt.subplots(4,3,figsize=(15,12))
for ax,nm in zip(axes.flat,[names[i] for i in order]):
    H=np.array(grid[nm]["H"]); ly=grid[nm]["layers"]; L=grid[nm]["L"]
    im=ax.imshow(H,aspect="auto",cmap="magma",vmin=0,vmax=vmax)
    ax.set_title(nm,fontsize=10); ax.set_yticks(range(4)); ax.set_yticklabels([c.split(" (")[0] for c in conds],fontsize=7)
    ax.set_xticks([0,len(ly)-1]); ax.set_xticklabels(["layer 0",f"layer {L-1}"],fontsize=7)
for ax in axes.flat[nM:]: ax.axis("off")
fig.suptitle("Per-model deconfounding maps — rows = confounds removed, columns = depth",fontsize=13)
fig.colorbar(im,ax=axes,label="signal (AUROC above baseline)",fraction=0.025); plt.savefig("fig_deconf_grid.png",dpi=130)

# ---- survival bars ----
fig,ax=plt.subplots(figsize=(9,5))
xs=np.arange(nM)
ax.bar(xs-0.2,rawbest[order],0.4,color="#888780",label="raw probe signal")
ax.bar(xs+0.2,deconfbest[order],0.4,color="#27ae60",label="signal surviving all deconfounding")
ax.set_xticks(xs); ax.set_xticklabels([names[i] for i in order],rotation=40,ha="right",fontsize=8)
ax.set_ylabel("signal (AUROC above baseline)"); ax.legend()
ax.set_title("How much probe signal is real vs. confound, per model"); plt.tight_layout(); plt.savefig("fig_deconf_survival.png",dpi=130)
print("saved fig_deconf_master.png, fig_deconf_grid.png, fig_deconf_survival.png")
print("DONE")
