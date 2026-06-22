import os
os.environ["HF_HUB_OFFLINE"] = "1"   # use only local cache; no token, no network
import torch, numpy as np, re, time, unicodedata, gc, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MODELS = [
 ("Qwen/Qwen2.5-1.5B-Instruct","Qwen2.5-1.5B","Alibaba"),
 ("Qwen/Qwen2.5-0.5B-Instruct","Qwen2.5-0.5B","Alibaba"),
 ("HuggingFaceTB/SmolLM2-1.7B-Instruct","SmolLM2-1.7B","HuggingFace"),
 ("HuggingFaceTB/SmolLM2-360M-Instruct","SmolLM2-360M","HuggingFace"),
 ("microsoft/Phi-3.5-mini-instruct","Phi-3.5-mini","Microsoft"),
 ("tiiuae/Falcon3-1B-Instruct","Falcon3-1B","TII"),
 ("allenai/OLMo-2-0425-1B-Instruct","OLMo-2-1B","AI2"),
 ("ibm-granite/granite-3.1-2b-instruct","Granite-3.1-2B","IBM"),
 ("stabilityai/stablelm-2-1_6b-chat","StableLM-2-1.6B","Stability"),
 ("google/gemma-2-2b-it","Gemma-2-2B","Google"),
 ("meta-llama/Llama-3.2-1B-Instruct","Llama-3.2-1B","Meta"),
 ("meta-llama/Llama-3.2-3B-Instruct","Llama-3.2-3B","Meta"),
]

capitals={"France":"Paris","Japan":"Tokyo","Brazil":"Brasilia","Egypt":"Cairo","Canada":"Ottawa","Australia":"Canberra","Turkey":"Ankara","Kazakhstan":"Astana","Nigeria":"Abuja","Myanmar":"Naypyidaw","Bhutan":"Thimphu","Nauru":"Yaren","Kiribati":"Tarawa","Tuvalu":"Funafuti","Palau":"Ngerulmud","Vanuatu":"Port Vila","Eritrea":"Asmara","Djibouti":"Djibouti","Comoros":"Moroni","Lesotho":"Maseru","Eswatini":"Mbabane","Suriname":"Paramaribo","Guyana":"Georgetown","Belize":"Belmopan","Brunei":"Bandar Seri Begawan","Mongolia":"Ulaanbaatar","Laos":"Vientiane","Cambodia":"Phnom Penh","Nepal":"Kathmandu","SriLanka":"Sri Jayawardenepura Kotte","Kyrgyzstan":"Bishkek","Tajikistan":"Dushanbe","Turkmenistan":"Ashgabat","Azerbaijan":"Baku","Armenia":"Yerevan","Georgia":"Tbilisi","Moldova":"Chisinau","Slovenia":"Ljubljana","Slovakia":"Bratislava","Croatia":"Zagreb","Serbia":"Belgrade","Albania":"Tirana","Macedonia":"Skopje","Montenegro":"Podgorica","Latvia":"Riga","Lithuania":"Vilnius","Estonia":"Tallinn","Iceland":"Reykjavik","Malta":"Valletta","Cyprus":"Nicosia","Senegal":"Dakar","Mali":"Bamako","Niger":"Niamey","Chad":"Ndjamena","Burkina Faso":"Ouagadougou","Benin":"Porto-Novo","Togo":"Lome","Gabon":"Libreville","Cameroon":"Yaounde","Angola":"Luanda","Zambia":"Lusaka","Malawi":"Lilongwe","Mozambique":"Maputo","Madagascar":"Antananarivo","Botswana":"Gaborone","Namibia":"Windhoek","Rwanda":"Kigali","Burundi":"Gitega","Uganda":"Kampala","Tanzania":"Dodoma","Paraguay":"Asuncion","Uruguay":"Montevideo","Bolivia":"Sucre","Ecuador":"Quito","Honduras":"Tegucigalpa","Nicaragua":"Managua","Panama":"Panama City","Jamaica":"Kingston","Bahamas":"Nassau","Qatar":"Doha","Oman":"Muscat","Yemen":"Sanaa","Jordan":"Amman","Lebanon":"Beirut","Bahrain":"Manama"}
elements=("Hydrogen Helium Lithium Beryllium Boron Carbon Nitrogen Oxygen Fluorine Neon Sodium Magnesium Aluminium Silicon Phosphorus Sulfur Chlorine Argon Potassium Calcium Scandium Titanium Vanadium Chromium Manganese Iron Cobalt Nickel Copper Zinc Gallium Germanium Arsenic Selenium Bromine Krypton Rubidium Strontium Yttrium Zirconium Niobium Molybdenum Technetium Ruthenium Rhodium Palladium Silver Cadmium Indium Tin Antimony Tellurium Iodine Xenon Caesium Barium Lanthanum Cerium Praseodymium Neodymium Promethium Samarium Europium Gadolinium Terbium Dysprosium Holmium Erbium Thulium Ytterbium Lutetium Hafnium Tantalum Tungsten Rhenium Osmium Iridium Platinum Gold Mercury Thallium Lead Bismuth Polonium Astatine Radon Francium Radium Actinium Thorium Protactinium Uranium Neptunium Plutonium Americium Curium Berkelium Californium Einsteinium Fermium").split()
items=[]
for c,cap in capitals.items(): items.append((f"What is the capital city of {c}? Reply with only the city name.",cap,"capital"))
for i,el in enumerate(elements): items.append((f"What is the atomic number of the element {el}? Reply with only the number.",str(i+1),"atomic"))

def nm(s): return "".join(ch for ch in unicodedata.normalize("NFKD",s) if not unicodedata.combining(ch)).lower().strip()
def grade(g,gold,kind):
    g=nm(g)
    if kind=="capital": return nm(gold).split()[0] in g
    n=re.findall(r"\d+",g); return bool(n) and n[0]==gold

def boot(y,s,B=3000):
    rng=np.random.default_rng(0); pos=np.where(y==1)[0]; neg=np.where(y==0)[0]; out=[]
    for _ in range(B):
        bi=np.concatenate([rng.choice(pos,len(pos),True),rng.choice(neg,len(neg),True)]); out.append(roc_auc_score(y[bi],s[bi]))
    return float(np.percentile(out,2.5)), float(np.percentile(out,97.5))

cv=StratifiedKFold(5,shuffle=True,random_state=0)
pipe=lambda: make_pipeline(StandardScaler(),LogisticRegression(C=0.05,max_iter=3000))

def run_model(repo, short, fam):
    tok=AutoTokenizer.from_pretrained(repo)
    model=AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()
    def prompt(u):
        try: return tok.apply_chat_template([{"role":"user","content":u}],tokenize=False,add_generation_prompt=True)
        except Exception: return u+"\n"
    X=[];y=[];ct=[];cvb=[]; t0=time.time()
    for k,(q,gold,kind) in enumerate(items):
        inp=tok(prompt(q),return_tensors="pt").to(DEV)
        with torch.no_grad():
            out=model(**inp,output_hidden_states=True)
            ptok=float(torch.softmax(out.logits[0,-1,:].float(),-1).max())
            gi=model.generate(**inp,max_new_tokens=12,do_sample=False,pad_token_id=tok.eos_token_id)
        a=tok.decode(gi[0][inp.input_ids.shape[1]:],skip_special_tokens=True).strip().splitlines(); ans=a[0] if a else ""
        ci=tok(prompt(f'Question: {q}\nAnswer given: "{ans}"\nHow confident are you (0-100) that this answer is correct? Reply with only an integer.'),return_tensors="pt").to(DEV)
        with torch.no_grad():
            cg=model.generate(**ci,max_new_tokens=6,do_sample=False,pad_token_id=tok.eos_token_id)
        m=re.search(r"\d{1,3}",tok.decode(cg[0][ci.input_ids.shape[1]:],skip_special_tokens=True))
        X.append(torch.stack([h[0,-1,:] for h in out.hidden_states]).float().cpu().numpy())
        y.append(int(grade(ans,gold,kind))); ct.append(ptok); cvb.append(min(100,int(m.group()))/100 if m else .5)
    X=np.array(X);y=np.array(y);ct=np.array(ct);cvb=np.array(cvb)
    np.savez(f"conf12_{short}.npz",X=X,y=y,conf_tok=ct,conf_verb=cvb)
    aucs=[roc_auc_score(y,cross_val_predict(pipe(),X[:,L,:],y,cv=cv,method="predict_proba")[:,1]) for L in range(X.shape[1])]
    bL=int(np.argmax(aucs)); probe_p=cross_val_predict(pipe(),X[:,bL,:],y,cv=cv,method="predict_proba")[:,1]
    res=dict(model=short,family=fam,n=int(len(y)),acc=float(y.mean()),
             verbalized=float(roc_auc_score(y,cvb)),token=float(roc_auc_score(y,ct)),probe=float(aucs[bL]),probe_layer=bL,
             verbalized_ci=boot(y,cvb),token_ci=boot(y,ct),probe_ci=boot(y,probe_p),
             secs=round(time.time()-t0,1))
    del model; gc.collect()
    if DEV=="mps": torch.mps.empty_cache()
    return res

results=[]
for repo,short,fam in MODELS:
    try:
        r=run_model(repo,short,fam); results.append(r)
        print(f"[done] {short:16s} acc={r['acc']:.0%} verbal={r['verbalized']:.3f} token={r['token']:.3f} probe={r['probe']:.3f} ({r['secs']}s)", flush=True)
    except Exception as e:
        print(f"[FAIL] {short}: {type(e).__name__}: {str(e)[:160]}", flush=True)
    json.dump(results, open("crossmodel12_results.json","w"), indent=2)

# ---- combined figure ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
results=sorted(results, key=lambda r:r["probe"])
ys=np.arange(len(results))
fig,ax=plt.subplots(figsize=(9, 0.5*len(results)+2))
ax.scatter([r["verbalized"] for r in results], ys, color="#c0392b", s=55, label="verbalized (what it SAYS)", zorder=3)
ax.scatter([r["token"] for r in results], ys, color="#2980b9", s=55, label="answer-token probability", zorder=3)
ax.scatter([r["probe"] for r in results], ys, color="#27ae60", s=55, label="linear probe (activations)", zorder=3)
for i,r in enumerate(results):
    ax.plot([r["verbalized_ci"][0],r["verbalized_ci"][1]],[i,i],color="#c0392b",alpha=.4,lw=2,zorder=2)
ax.axvline(0.5,color="k",ls="--",lw=1); ax.text(0.5,len(results)-0.3,"chance",ha="center",fontsize=8)
ax.set_yticks(ys); ax.set_yticklabels([f"{r['model']} ({r['family']}, acc {r['acc']:.0%})" for r in results], fontsize=8)
ax.set_xlim(0.3,1.0); ax.set_xlabel("AUROC: predicts its own correctness")
ax.set_title(f"{len(results)} model families: verbalized confidence hugs chance, internals don't")
ax.legend(loc="lower right", fontsize=8); plt.tight_layout(); plt.savefig("crossmodel12.png",dpi=130)
print("\nsaved crossmodel12.png and crossmodel12_results.json", flush=True)
print(f"\n{'model':16s}{'fam':12s}{'acc':>6s}{'verbal':>8s}{'token':>7s}{'probe':>7s}")
for r in results: print(f"{r['model']:16s}{r['family']:12s}{r['acc']:6.0%}{r['verbalized']:8.3f}{r['token']:7.3f}{r['probe']:7.3f}")
print("ALL DONE", flush=True)
