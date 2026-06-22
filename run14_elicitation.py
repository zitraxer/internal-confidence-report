import os
os.environ["HF_HUB_OFFLINE"]="1"
import torch, numpy as np, re, time, unicodedata, gc, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score
DEV="mps" if torch.backends.mps.is_available() else "cpu"

# error-rich models only (enough wrong answers for a stable AUROC)
MODELS=[("HuggingFaceTB/SmolLM2-1.7B-Instruct","SmolLM2-1.7B"),
        ("stabilityai/stablelm-2-1_6b-chat","StableLM-2-1.6B"),
        ("Qwen/Qwen2.5-0.5B-Instruct","Qwen2.5-0.5B"),
        ("allenai/OLMo-2-0425-1B-Instruct","OLMo-2-1B")]

capitals={"France":"Paris","Japan":"Tokyo","Brazil":"Brasilia","Egypt":"Cairo","Canada":"Ottawa","Australia":"Canberra","Turkey":"Ankara","Kazakhstan":"Astana","Nigeria":"Abuja","Myanmar":"Naypyidaw","Bhutan":"Thimphu","Nauru":"Yaren","Kiribati":"Tarawa","Tuvalu":"Funafuti","Palau":"Ngerulmud","Vanuatu":"Port Vila","Eritrea":"Asmara","Djibouti":"Djibouti","Comoros":"Moroni","Lesotho":"Maseru","Eswatini":"Mbabane","Suriname":"Paramaribo","Guyana":"Georgetown","Belize":"Belmopan","Brunei":"Bandar Seri Begawan","Mongolia":"Ulaanbaatar","Laos":"Vientiane","Cambodia":"Phnom Penh","Nepal":"Kathmandu","SriLanka":"Sri Jayawardenepura Kotte","Kyrgyzstan":"Bishkek","Tajikistan":"Dushanbe","Turkmenistan":"Ashgabat","Azerbaijan":"Baku","Armenia":"Yerevan","Georgia":"Tbilisi","Moldova":"Chisinau","Slovenia":"Ljubljana","Slovakia":"Bratislava","Croatia":"Zagreb","Serbia":"Belgrade","Albania":"Tirana","Macedonia":"Skopje","Montenegro":"Podgorica","Latvia":"Riga","Lithuania":"Vilnius","Estonia":"Tallinn","Iceland":"Reykjavik","Malta":"Valletta","Cyprus":"Nicosia","Senegal":"Dakar","Mali":"Bamako","Niger":"Niamey","Chad":"Ndjamena","Burkina Faso":"Ouagadougou","Benin":"Porto-Novo","Togo":"Lome","Gabon":"Libreville","Cameroon":"Yaounde","Angola":"Luanda","Zambia":"Lusaka","Malawi":"Lilongwe","Mozambique":"Maputo","Madagascar":"Antananarivo","Botswana":"Gaborone","Namibia":"Windhoek","Rwanda":"Kigali","Burundi":"Gitega","Uganda":"Kampala","Tanzania":"Dodoma","Paraguay":"Asuncion","Uruguay":"Montevideo","Bolivia":"Sucre","Ecuador":"Quito","Honduras":"Tegucigalpa","Nicaragua":"Managua","Panama":"Panama City","Jamaica":"Kingston","Bahamas":"Nassau","Qatar":"Doha","Oman":"Muscat","Yemen":"Sanaa","Jordan":"Amman","Lebanon":"Beirut","Bahrain":"Manama"}
elements=("Hydrogen Helium Lithium Beryllium Boron Carbon Nitrogen Oxygen Fluorine Neon Sodium Magnesium Aluminium Silicon Phosphorus Sulfur Chlorine Argon Potassium Calcium Scandium Titanium Vanadium Chromium Manganese Iron Cobalt Nickel Copper Zinc Gallium Germanium Arsenic Selenium Bromine Krypton Rubidium Strontium Yttrium Zirconium Niobium Molybdenum Technetium Ruthenium Rhodium Palladium Silver Cadmium Indium Tin Antimony Tellurium Iodine Xenon Caesium Barium Lanthanum Cerium Praseodymium Neodymium Promethium Samarium Europium Gadolinium Terbium Dysprosium Holmium Erbium Thulium Ytterbium Lutetium Hafnium Tantalum Tungsten Rhenium Osmium Iridium Platinum Gold Mercury Thallium Lead Bismuth Polonium Astatine Radon Francium Radium Actinium Thorium Protactinium Uranium Neptunium Plutonium Americium Curium Berkelium Californium Einsteinium Fermium").split()
items=[]
for c,cap in capitals.items(): items.append((f"What is the capital city of {c}?",cap,"capital"))
for i,el in enumerate(elements): items.append((f"What is the atomic number of the element {el}?",str(i+1),"atomic"))

def nm(s): return "".join(ch for ch in unicodedata.normalize("NFKD",s) if not unicodedata.combining(ch)).lower().strip()
def grade(g,gold,kind):
    g=nm(g)
    if kind=="capital": return nm(gold).split()[0] in g
    n=re.findall(r"\d+",g); return bool(n) and n[0]==gold

# four ways to ASK for confidence (steelman: give the model its best shot)
def P_pct(q,a):  return f'Question: {q}\nProposed answer: "{a}"\nOn a scale of 0 to 100, how confident are you this answer is correct? Reply with only the number.'
def P_true(q,a): return f'Question: {q}\nProposed answer: "{a}"\nWhat is the probability (0.00 to 1.00) that this answer is correct? Reply with only the probability.'
def P_cat(q,a):  return f'Question: {q}\nProposed answer: "{a}"\nHow certain are you this is correct? Reply with exactly one of: very low, low, medium, high, very high.'
def P_yn(q,a):   return f'Question: {q}\nProposed answer: "{a}"\nAre you confident this answer is correct? Reply with only yes or no.'

def parse_pct(t):
    m=re.search(r"\d{1,3}",t); return min(100,int(m.group()))/100 if m else .5
def parse_true(t):
    m=re.search(r"0?\.\d+|\d{1,3}",t)
    if not m: return .5
    v=float(m.group()); return min(1.0,v) if v<=1 else min(100,v)/100
def parse_cat(t):
    t=t.lower()
    if "very high" in t: return .9
    if "very low" in t: return .1
    if "high" in t: return .7
    if "medium" in t: return .5
    if "low" in t: return .3
    return .5
def parse_yn(t):
    t=t.lower(); return 1.0 if ("yes" in t and "no" not in t[:t.find("yes")+3]) else (0.0 if "no" in t else .5)

def run(repo, short):
    tok=AutoTokenizer.from_pretrained(repo)
    model=AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV).eval()
    def chat(u): return tok.apply_chat_template([{"role":"user","content":u}],tokenize=False,add_generation_prompt=True)
    def gen(prompt,n=8):
        inp=tok(chat(prompt),return_tensors="pt").to(DEV)
        with torch.no_grad(): g=model.generate(**inp,max_new_tokens=n,do_sample=False,pad_token_id=tok.eos_token_id)
        return tok.decode(g[0][inp.input_ids.shape[1]:],skip_special_tokens=True).strip(), inp
    y=[];tok_c=[];pct=[];ptr=[];cat=[];yn_=[]
    t0=time.time()
    for k,(q,gold,kind) in enumerate(items):
        ainp=tok(chat(q+" Reply with only the answer."),return_tensors="pt").to(DEV)
        with torch.no_grad():
            out=model(**ainp); ptok=float(torch.softmax(out.logits[0,-1,:].float(),-1).max())
            ga=model.generate(**ainp,max_new_tokens=12,do_sample=False,pad_token_id=tok.eos_token_id)
        a=tok.decode(ga[0][ainp.input_ids.shape[1]:],skip_special_tokens=True).strip().splitlines(); ans=a[0] if a else ""
        y.append(int(grade(ans,gold,kind))); tok_c.append(ptok)
        pct.append(parse_pct(gen(P_pct(q,ans),6)[0]))
        ptr.append(parse_true(gen(P_true(q,ans),8)[0]))
        cat.append(parse_cat(gen(P_cat(q,ans),6)[0]))
        yn_.append(parse_yn(gen(P_yn(q,ans),4)[0]))
        if (k+1)%60==0: print(f"  [{short}] {k+1}/{len(items)} ({time.time()-t0:.0f}s)",flush=True)
    y=np.array(y)
    def safe(s):
        s=np.array(s)
        try: return float(roc_auc_score(y,s))
        except Exception: return float("nan")
    r=dict(model=short,n=int(len(y)),err=int((y==0).sum()),
           internal_token=safe(tok_c),
           verbal_pct=safe(pct),verbal_Ptrue=safe(ptr),verbal_categorical=safe(cat),verbal_yesno=safe(yn_))
    del model; gc.collect()
    if DEV=="mps": torch.mps.empty_cache()
    return r

res=[]
for repo,short in MODELS:
    try:
        r=run(repo,short); res.append(r)
        print(f"[done] {short:16s} err={r['err']:3d} | internal(token)={r['internal_token']:.3f} | "
              f"verbal: pct={r['verbal_pct']:.3f} Ptrue={r['verbal_Ptrue']:.3f} categ={r['verbal_categorical']:.3f} yes/no={r['verbal_yesno']:.3f}",flush=True)
    except Exception as e:
        print(f"[FAIL] {short}: {type(e).__name__}: {str(e)[:160]}",flush=True)
    json.dump(res,open("elicitation_results.json","w"),indent=2,default=float)

if res:
    bestverbal=[max(r['verbal_pct'],r['verbal_Ptrue'],r['verbal_categorical'],r['verbal_yesno']) for r in res]
    print(f"\nmean BEST-of-4 verbalized AUROC = {np.mean(bestverbal):.3f} | mean internal(token) = {np.mean([r['internal_token'] for r in res]):.3f}")
    print("If even the best elicitation stays well below the internal signal, 'words are unreliable' is not a prompt artifact.")
print("DONE",flush=True)
