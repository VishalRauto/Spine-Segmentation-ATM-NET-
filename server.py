import sys, os, io, json, base64, time, uuid, threading, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string
import numpy as np
import cv2

BASE       = Path(__file__).parent
CKPT_BEST  = BASE / "outputs/gpu_run/best_model.pth"
CKPT_LAST  = BASE / "outputs/gpu_run/last_model.pth"
CKPT_CPU   = BASE / "outputs/high_perf_run/best_model.pth"
HIST_GPU   = BASE / "outputs/gpu_run/history.json"
HIST_CPU   = BASE / "outputs/high_perf_run/history.json"
UPLOAD_DIR = BASE / "outputs/uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def _pick_ckpt():
    """Pick the best compatible checkpoint — prefers highest epoch count (most trained)."""
    import torch
    candidates = [
        BASE / "outputs/gpu_run/last_model.pth",        # epoch 138 — most trained local
        BASE / "outputs/gpu_run/best_model.pth",         # may be overwritten by kaggle
        CKPT_CPU,
    ]
    best_path, best_ep, best_dice = None, -1, -1.0
    for p in candidates:
        if not p.exists():
            continue
        try:
            c = torch.load(str(p), map_location="cpu")
            keys = list(c.get("model_state_dict", {}).keys())
            # Full compatibility: needs enc conv keys AND sa.conv keys
            has_enc_conv = any(k == "e1.conv.0.weight" for k in keys)
            has_sa_conv  = any("sa.conv.0.weight" in k  for k in keys)
            if not (has_enc_conv and has_sa_conv):
                print(f"  [ckpt] Skipping {p.name} — incompatible")
                continue
            ep   = c.get("epoch", 0)
            dice = c.get("best_dice", 0.0)
            print(f"  [ckpt] {p.name}: epoch={ep} dice={dice:.4f} ✓")
            # Prefer most-trained (highest epoch) — more training = better feature maps
            if ep > best_ep:
                best_ep, best_dice, best_path = ep, dice, p
        except Exception as e:
            print(f"  [ckpt] {p.name}: error — {e}")
    if best_path:
        print(f"  [ckpt] Using: {best_path.name} (epoch={best_ep}, dice={best_dice:.4f})")
    else:
        print("  [ckpt] WARNING: No compatible checkpoint found.")
    return best_path

CKPT = _pick_ckpt()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

_model = None
_model_lock = threading.Lock()
_device = None

SPIDER_TO_ATMNET = {**{i: i for i in range(1, 9)}, 100: 9,
                    **{201+i: 10+i for i in range(8)}}
CLASS_NAMES = {
    0:"Background",1:"Vertebra-1",2:"Vertebra-2",3:"Vertebra-3",
    4:"Vertebra-4",5:"Vertebra-5",6:"Vertebra-6",7:"Vertebra-7",
    8:"Vertebra-8",9:"Sacrum",
    10:"IVD-1",11:"IVD-2",12:"IVD-3",13:"IVD-4",
    14:"IVD-5",15:"IVD-6",16:"IVD-7",17:"IVD-8",18:"Spinal Canal",
}
COLORS = {
    0:(0,0,0),1:(255,50,50),2:(255,120,50),3:(255,200,50),
    4:(200,255,50),5:(100,255,50),6:(50,255,100),7:(50,255,200),
    8:(50,200,255),9:(50,100,255),10:(150,50,255),11:(255,50,200),
    12:(255,50,100),13:(200,100,255),14:(100,200,255),15:(255,150,50),
    16:(50,255,150),17:(150,255,50),18:(220,220,220),
}
NUM_CLASSES = 19
VERT_CLASSES = list(range(1,9))
IVD_CLASSES  = list(range(10,18))

def remap(m):
    out = np.zeros_like(m, dtype=np.int64)
    for s,d in SPIDER_TO_ATMNET.items(): out[m==s]=d
    return out

def get_model():
    global _model, _device
    with _model_lock:
        if _model is not None: return _model, _device
        import torch, torch.nn as nn, torch.nn.functional as F
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Architecture MUST match train_best.py exactly ──────────────
        class CA(nn.Module):
            def __init__(self,ch,r=8):
                super().__init__()
                r=max(1,ch//r)
                self.avg=nn.AdaptiveAvgPool2d(1); self.max=nn.AdaptiveMaxPool2d(1)
                self.fc=nn.Sequential(nn.Flatten(),nn.Linear(ch,r),nn.ReLU(True),nn.Linear(r,ch),nn.Sigmoid())
            def forward(self,x):
                a=self.fc(self.avg(x))+self.fc(self.max(x))
                return x*a.clamp(0,1).view(x.shape[0],-1,1,1)

        class SA(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv=nn.Sequential(nn.Conv2d(2,1,7,padding=3,bias=False),nn.BatchNorm2d(1),nn.Sigmoid())
            def forward(self,x):
                return x*self.conv(torch.cat([x.mean(1,keepdim=True),x.max(1,keepdim=True)[0]],1))

        class RB(nn.Module):
            def __init__(self,ch):
                super().__init__()
                self.net=nn.Sequential(nn.Conv2d(ch,ch,3,1,1,bias=False),nn.BatchNorm2d(ch),nn.ReLU(True),
                                       nn.Conv2d(ch,ch,3,1,1,bias=False),nn.BatchNorm2d(ch))
                self.ca=CA(ch); self.sa=SA(); self.act=nn.ReLU(True)
            def forward(self,x): return self.act(self.sa(self.ca(self.net(x)))+x)

        class Enc(nn.Module):
            # drop param required — matches train_best.py Enc(ci, co, drop=0.0)
            def __init__(self,ci,co,drop=0.0):
                super().__init__()
                self.conv=nn.Sequential(nn.Conv2d(ci,co,3,1,1,bias=False),nn.BatchNorm2d(co),nn.ReLU(True),
                                        nn.Conv2d(co,co,3,1,1,bias=False),nn.BatchNorm2d(co),nn.ReLU(True))
                self.res=RB(co)
                self.drop=nn.Dropout2d(drop) if drop>0 else nn.Identity()
            def forward(self,x): return self.drop(self.res(self.conv(x)))

        class ResUNet(nn.Module):
            # Exact replica of train_best.py ResUNet — includes ds3, ds2, aux head
            def __init__(self,b=32,nc=NUM_CLASSES,drop=0.25):
                super().__init__()
                self.e1=Enc(1,b); self.e2=Enc(b,b*2,drop*0.3)
                self.e3=Enc(b*2,b*4,drop*0.6); self.e4=Enc(b*4,b*8,drop*0.8)
                self.bn=nn.Sequential(Enc(b*8,b*16,drop),nn.Dropout2d(drop))
                self.pool=nn.MaxPool2d(2)
                self.u4=nn.ConvTranspose2d(b*16,b*8,2,2); self.d4=Enc(b*16,b*8,drop*0.4)
                self.u3=nn.ConvTranspose2d(b*8,b*4,2,2);  self.d3=Enc(b*8,b*4,drop*0.2)
                self.u2=nn.ConvTranspose2d(b*4,b*2,2,2);  self.d2=Enc(b*4,b*2)
                self.u1=nn.ConvTranspose2d(b*2,b,2,2);    self.d1=Enc(b*2,b)
                self.ds3=nn.Conv2d(b*4,nc,1); self.ds2=nn.Conv2d(b*2,nc,1); self.out=nn.Conv2d(b,nc,1)
                self.aux=nn.Sequential(nn.Conv2d(b,b,3,1,1,bias=False),nn.BatchNorm2d(b),nn.ReLU(True),nn.Conv2d(b,nc,1))
            def forward(self,x):
                # Inference mode — return only the main output (no aux heads)
                sz=x.shape[2:]
                e1=self.e1(x); e2=self.e2(self.pool(e1)); e3=self.e3(self.pool(e2)); e4=self.e4(self.pool(e3))
                d=self.bn(self.pool(e4))
                d=self.d4(torch.cat([self.u4(d),e4],1))
                d=self.d3(torch.cat([self.u3(d),e3],1))
                d=self.d2(torch.cat([self.u2(d),e2],1))
                d=self.d1(torch.cat([self.u1(d),e1],1))
                return self.out(d)   # only main head at inference

        _model = ResUNet(b=32, nc=NUM_CLASSES, drop=0.25).to(_device)

        if CKPT and CKPT.exists():
            ckpt = torch.load(str(CKPT), map_location=_device)
            missing, unexpected = _model.load_state_dict(ckpt["model_state_dict"], strict=False)
            ep   = ckpt.get("epoch", "?")
            dice = ckpt.get("best_dice", 0.0)
            print(f"Loaded checkpoint: epoch={ep}, dice={dice:.4f}")
            if missing:    print(f"  Missing keys  : {len(missing)}  (aux/ds heads not needed at inference)")
            if unexpected: print(f"  Unexpected keys: {len(unexpected)}")
        else:
            print("WARNING: No compatible checkpoint found — model using random weights!")
            print("  Run training first: python scripts/train_best.py")

        _model.eval()
        return _model, _device

def arr_to_b64(arr):
    if arr.ndim==2:
        if arr.max()<=1: arr=(arr*255).astype(np.uint8)
        arr=cv2.cvtColor(arr.astype(np.uint8),cv2.COLOR_GRAY2RGB)
    _,buf=cv2.imencode(".png",cv2.cvtColor(arr,cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode()

def colorize(mask):
    rgb=np.zeros((*mask.shape,3),dtype=np.uint8)
    for c,col in COLORS.items(): rgb[mask==c]=col
    return rgb

def run_pred(file_bytes,filename):
    import torch,torch.nn.functional as F
    model,device=get_model()
    ext="".join(Path(filename).suffixes).lower()
    size=192
    if ext in {".png",".jpg",".jpeg"}:
        npa=np.frombuffer(file_bytes,np.uint8)
        img=cv2.imdecode(npa,cv2.IMREAD_GRAYSCALE).astype(np.float32)
        slices=[img]
    else:
        tmp=UPLOAD_DIR/f"{uuid.uuid4()}{ext}"
        tmp.write_bytes(file_bytes)
        try:
            import SimpleITK as sitk
            vol=sitk.GetArrayFromImage(sitk.ReadImage(str(tmp))).astype(np.float32)
            n=vol.shape[0]; lo,hi=int(n*0.15),int(n*0.85)
            step=max(1,(hi-lo)//6); slices=[vol[i] for i in range(lo,hi,step)][:6]
        finally: tmp.unlink(missing_ok=True)

    preds=[]
    for sl in slices:
        p1,p99=np.percentile(sl,[0.5,99.5])
        img_n=np.clip((sl-p1)/(p99-p1+1e-8),0,1).astype(np.float32)
        img_r=cv2.resize(img_n,(size,size),interpolation=cv2.INTER_LINEAR)
        t=torch.from_numpy(img_r[None,None]).float().to(device)
        with torch.no_grad():
            pr=F.softmax(model(t),1)
            pr2=F.softmax(model(torch.flip(t,[-1])),1)
            pr2=torch.flip(pr2,[-1])
            avg=(pr+pr2)/2

        # ── Background-bias correction ──────────────────────────────
        # Models trained with class imbalance often over-predict background.
        # Fix: suppress background by subtracting a bias so that fg classes
        # compete fairly. We subtract enough from bg to bring it level with
        # the strongest foreground class, capped so we don't hallucinate.
        avg_np = avg.squeeze(0).cpu().numpy()   # [C, H, W]
        bg_prob  = avg_np[0]                     # background channel
        fg_probs = avg_np[1:]                    # all foreground channels
        fg_max   = fg_probs.max(0)               # best fg probability per pixel

        # Only suppress bg where fg has reasonable signal (>1% probability)
        # Bias = how much bg exceeds the best fg class
        bg_bias  = np.clip(bg_prob - fg_max - 0.02, 0, None)  # don't over-suppress
        avg_np[0] = bg_prob - bg_bias            # bring bg down to fg level

        # Re-normalise to sum=1
        avg_np = avg_np / (avg_np.sum(0, keepdims=True) + 1e-8)

        pred = avg_np.argmax(0).astype(np.int32)
        preds.append((img_r, pred, avg_np))

    mid=len(preds)//2
    img_pre,pred,prob_arr=preds[mid]

    cls_dist={}; detected=[]
    for c in range(1,NUM_CLASSES):
        px=int((pred==c).sum())
        if px>30:
            nm=CLASS_NAMES[c]; pct=round(px/pred.size*100,2)
            cls_dist[nm]={"pixels":px,"percent":pct}
            detected.append(nm)

    ivd_p=[float(prob_arr[c].max()) for c in IVD_CLASSES]
    avg_ivd=float(np.mean(ivd_p)) if ivd_p else 0
    if avg_ivd>0.75: disease,severity,conf="Normal","None",round(avg_ivd*100,1)
    elif avg_ivd>0.5: disease,severity,conf="Disc Bulge","Mild",round(avg_ivd*100,1)
    elif avg_ivd>0.3: disease,severity,conf="Disc Degeneration","Moderate",round(avg_ivd*100,1)
    else: disease,severity,conf="Disc Herniation","Severe",round((1-avg_ivd)*100,1)
    pfirrmann=round(5-avg_ivd*4,1)

    img_u8=(np.clip(img_pre,0,1)*255).astype(np.uint8)
    img_rgb=cv2.cvtColor(img_u8,cv2.COLOR_GRAY2RGB)
    mask_rgb=colorize(pred)
    fg=(pred>0).astype(np.float32)[...,np.newaxis]
    blend=(img_rgb*(1-0.6*fg)+mask_rgb*0.6*fg).astype(np.uint8)

    report=f"LUMBAR SPINE MRI REPORT - ATM-Net++\n{'='*40}\nDiagnosis   : {disease}\nSeverity    : {severity}\nConfidence  : {conf}%\nPfirrmann   : {pfirrmann}/5\n\nDetected Structures:\n"
    for s in detected: report+=f"  * {s}\n"
    report+=f"\nFindings: {disease} with {severity.lower()} severity.\nPfirrmann {pfirrmann}: {'normal.' if pfirrmann<=2 else 'early degeneration.' if pfirrmann<=3 else 'advanced degeneration.'}\n\nWARNING: AI-generated. Requires radiologist review."
    return {"num_slices":len(slices),"detected_structures":detected,"class_distribution":cls_dist,
            "disease":disease,"severity":severity,"confidence":conf,"pfirrmann_grade":pfirrmann,
            "image_b64":arr_to_b64((img_pre*255).astype(np.uint8)),
            "overlay_b64":arr_to_b64(blend),"mask_b64":arr_to_b64(mask_rgb),"report":report}

@app.route("/")
def index(): return render_template_string(HTML)

@app.route("/predict",methods=["POST"])
def predict():
    if "file" not in request.files: return jsonify({"error":"No file"}),400
    f=request.files["file"]
    try:
        t0=time.time(); res=run_pred(f.read(),f.filename)
        res["inference_ms"]=round((time.time()-t0)*1000); return jsonify(res)
    except Exception as e:
        import traceback; return jsonify({"error":str(e),"trace":traceback.format_exc()}),500

@app.route("/training")
def training():
    hist=[]
    for hf in [HIST_GPU,HIST_CPU]:
        if hf.exists():
            try: hist=json.load(open(hf)); break
            except: pass
    ckpts={}
    for lbl,ck in [("Best",CKPT_BEST),("Last",CKPT_LAST),("CPU",CKPT_CPU)]:
        if ck.exists():
            try:
                import torch; c=torch.load(str(ck),map_location="cpu")
                ckpts[lbl]={"epoch":c.get("epoch","?"),"best_dice":round(c.get("best_dice",0),4)}
            except: pass
    return jsonify({"history":hist[-60:],"checkpoints":ckpts})

@app.route("/health")
def health():
    import torch
    info={"status":"running","gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}
    if CKPT and CKPT.exists():
        try:
            c=torch.load(str(CKPT),map_location="cpu")
            info["checkpoint"]=f"epoch {c.get('epoch','?')} | dice={c.get('best_dice',0):.4f}"
            info["checkpoint_file"]=CKPT.name
        except: pass
    return jsonify(info)

HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ATM-Net++ | Spine MRI Analysis</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0f1117;color:#e2e8f0;min-height:100vh}
.nav{background:#1a1f2e;border-bottom:1px solid #2d3748;padding:14px 24px;display:flex;align-items:center;gap:16px}
.logo{font-size:20px;font-weight:700;color:#63b3ed}.logo span{color:#68d391}
.nav-tabs{display:flex;gap:4px;margin-left:auto}
.tab-btn{padding:7px 18px;border-radius:8px;border:none;cursor:pointer;font-size:13px;font-weight:500;background:transparent;color:#a0aec0;transition:.2s}
.tab-btn.active,.tab-btn:hover{background:#2d3748;color:#e2e8f0}
.container{max-width:1100px;margin:0 auto;padding:24px}
.section{display:none}.section.active{display:block}
.drop-zone{border:2px dashed #4a5568;border-radius:16px;padding:48px;text-align:center;cursor:pointer;transition:.2s;background:#1a1f2e}
.drop-zone:hover,.drop-zone.drag{border-color:#63b3ed;background:#1e2a3a}
.drop-zone .icon{font-size:48px;margin-bottom:12px}
.file-info{background:#2d3748;border-radius:8px;padding:10px 14px;margin-top:12px;font-size:13px;color:#68d391;display:none}
.form-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:16px}
.form-group{display:flex;flex-direction:column;gap:4px}
.form-group label{font-size:12px;color:#a0aec0;font-weight:500}
.form-group input,.form-group select,.form-group textarea{background:#2d3748;border:1px solid #4a5568;border-radius:8px;padding:8px 12px;color:#e2e8f0;font-size:13px;outline:none;transition:.2s}
.form-group input:focus,.form-group select:focus,.form-group textarea:focus{border-color:#63b3ed}
.form-group textarea{resize:vertical;min-height:70px}
.full-width{grid-column:1/-1}
.btn{padding:12px 24px;border-radius:10px;border:none;cursor:pointer;font-size:14px;font-weight:600;transition:.2s;display:inline-flex;align-items:center;gap:8px}
.btn-primary{background:#3182ce;color:white}.btn-primary:hover{background:#2b6cb0}
.btn-primary:disabled{background:#4a5568;cursor:not-allowed}
.btn-outline{background:transparent;border:1px solid #4a5568;color:#a0aec0}.btn-outline:hover{border-color:#63b3ed;color:#63b3ed}
.results-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-top:20px}
.result-card{background:#1a1f2e;border-radius:12px;padding:16px;border:1px solid #2d3748}
.result-card h3{font-size:12px;color:#a0aec0;font-weight:500;margin-bottom:12px;text-transform:uppercase;letter-spacing:.5px}
.result-card img{width:100%;border-radius:8px;border:1px solid #2d3748}
.diag-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}
.diag-card{background:#1a1f2e;border-radius:12px;padding:16px;border:1px solid #2d3748;text-align:center}
.diag-label{font-size:10px;color:#718096;margin-bottom:6px;text-transform:uppercase;letter-spacing:.5px}
.diag-val{font-size:18px;font-weight:700;color:#63b3ed}
.metric-row{display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid #2d3748;font-size:13px}
.metric-row:last-child{border:none}
.struct-tags{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.tag{padding:3px 10px;background:#2d3748;border-radius:20px;font-size:11px;color:#a0aec0}
.tag.vert{color:#68d391;background:#1c3a2a}.tag.ivd{color:#63b3ed;background:#1a2a3a}.tag.other{color:#f6ad55;background:#3a2a1a}
.report-box{background:#1a1f2e;border:1px solid #2d3748;border-radius:12px;padding:16px;font-family:monospace;font-size:12px;color:#a0aec0;white-space:pre-wrap;max-height:280px;overflow-y:auto;margin-top:16px}
.train-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px}
.stat-card{background:#1a1f2e;border-radius:12px;padding:16px;border:1px solid #2d3748;text-align:center}
.stat-val{font-size:28px;font-weight:700;color:#63b3ed}
.stat-label{font-size:11px;color:#718096;margin-top:4px}
#chartArea{background:#1a1f2e;border-radius:12px;padding:20px;border:1px solid #2d3748;margin-bottom:16px}
.epoch-table{width:100%;border-collapse:collapse;font-size:12px}
.epoch-table th{background:#2d3748;padding:8px 12px;text-align:left;color:#a0aec0;font-weight:500}
.epoch-table td{padding:7px 12px;border-bottom:1px solid #2d3748}
.epoch-table tr.best td{color:#68d391}
.spinner{display:inline-block;width:16px;height:16px;border:2px solid rgba(255,255,255,.2);border-top-color:#63b3ed;border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:999;align-items:center;justify-content:center;flex-direction:column;gap:16px;font-size:16px}
.overlay.on{display:flex}
@media(max-width:768px){.results-grid,.form-grid,.train-grid,.diag-grid{grid-template-columns:1fr}.full-width{grid-column:1}}
</style></head><body>
<div class="overlay" id="ov"><div class="spinner" style="width:40px;height:40px;border-width:4px"></div><div id="ovTxt" style="color:#e2e8f0">Analyzing MRI...</div></div>
<nav class="nav">
  <div class="logo">ATM-Net<span>++</span></div>
  <span style="color:#4a5568;font-size:13px">Anatomy-Aware Spine MRI Analysis</span>
  <div class="nav-tabs">
    <button class="tab-btn active" onclick="showTab('predict',this)">🔬 Predict</button>
    <button class="tab-btn" onclick="showTab('training',this)">📈 Training</button>
    <button class="tab-btn" onclick="showTab('about',this)">ℹ️ About</button>
  </div>
</nav>

<!-- PREDICT -->
<div class="section active" id="s-predict"><div class="container">
  <h2 style="font-size:22px;font-weight:700;margin-bottom:4px">Lumbar Spine MRI Analysis</h2>
  <p style="color:#718096;font-size:14px;margin-bottom:24px">Upload MRI → get segmentation, disease prediction, clinical report</p>
  <div class="drop-zone" id="dz" onclick="document.getElementById('fi').click()">
    <div class="icon">🩻</div>
    <h3>Drop your MRI file here</h3>
    <p>Supports: .mha .mhd .nii .nii.gz .dcm .png .jpg</p>
    <input type="file" id="fi" style="display:none" accept=".mha,.mhd,.nii,.gz,.dcm,.png,.jpg,.jpeg" onchange="setFile(this.files[0])">
  </div>
  <div class="file-info" id="finfo"></div>
  <div style="margin-top:18px"><div style="font-size:12px;color:#718096;font-weight:500;margin-bottom:8px;text-transform:uppercase;letter-spacing:.5px">Patient Details (optional)</div>
  <div class="form-grid">
    <div class="form-group"><label>Sex</label><select id="sex"><option value="">—</option><option value="F">Female</option><option value="M">Male</option></select></div>
    <div class="form-group"><label>Age (years)</label><input type="number" id="age" placeholder="e.g. 55" min="0" max="120"></div>
    <div class="form-group"><label>Height (cm)</label><input type="number" id="ht" placeholder="e.g. 170"></div>
    <div class="form-group"><label>Weight (kg)</label><input type="number" id="wt" placeholder="e.g. 75"></div>
    <div class="form-group full-width"><label>Radiology Report (optional)</label>
    <textarea id="rtext" placeholder="e.g. Posterior disc bulge at L4-L5 causing mild spinal canal stenosis..."></textarea></div>
  </div></div>
  <div style="margin-top:18px;display:flex;gap:12px">
    <button class="btn btn-primary" id="aBtn" onclick="analyze()" disabled>🔬 Analyze MRI</button>
    <button class="btn btn-outline" onclick="clearAll()">✕ Clear</button>
  </div>
  <div id="results" style="display:none;margin-top:32px">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
      <h3 style="font-size:18px;font-weight:700">Analysis Results</h3>
      <span id="itime" style="font-size:12px;color:#718096"></span>
    </div>
    <div class="diag-grid">
      <div class="diag-card"><div class="diag-label">Diagnosis</div><div class="diag-val" id="r-dis">—</div></div>
      <div class="diag-card"><div class="diag-label">Severity</div><div class="diag-val" id="r-sev">—</div></div>
      <div class="diag-card"><div class="diag-label">Confidence</div><div class="diag-val" id="r-con" style="color:#68d391">—</div></div>
      <div class="diag-card"><div class="diag-label">Pfirrmann Grade</div><div class="diag-val" id="r-pfi" style="color:#f6ad55">—</div></div>
    </div>
    <div class="results-grid">
      <div class="result-card"><h3>MRI Input</h3><img id="i-orig" src="" alt="Original"></div>
      <div class="result-card"><h3>Segmentation Overlay</h3><img id="i-over" src="" alt="Overlay"></div>
      <div class="result-card"><h3>Segmentation Mask</h3><img id="i-mask" src="" alt="Mask"></div>
    </div>
    <div class="result-card" style="margin-top:16px"><h3>Detected Structures</h3><div class="struct-tags" id="stags"></div></div>
    <div class="result-card" style="margin-top:16px"><h3>Class Distribution</h3><div id="cdist"></div></div>
    <div style="margin-top:16px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <div style="font-size:12px;font-weight:600;color:#a0aec0;text-transform:uppercase;letter-spacing:.5px">Clinical Report</div>
        <button class="btn btn-outline" style="padding:5px 14px;font-size:12px" onclick="dlReport()">⬇ Download</button>
      </div>
      <div class="report-box" id="rbox"></div>
    </div>
    <div style="margin-top:14px;padding:12px 16px;background:#2d1515;border-radius:10px;border:1px solid #742a2a;font-size:12px;color:#fc8181">
      ⚠️ AI-generated. Must be reviewed by a qualified radiologist before any clinical use.
    </div>
  </div>
</div></div>

<!-- TRAINING -->
<div class="section" id="s-training"><div class="container">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px">
    <div><h2 style="font-size:22px;font-weight:700">Training Monitor</h2>
    <p style="color:#718096;font-size:14px;margin-top:4px">Live progress on SPIDER dataset — RTX 3050 | ResUNet+CBAM | 192×192</p></div>
    <button class="btn btn-outline" style="font-size:13px" onclick="loadTrain()">🔄 Refresh</button>
  </div>
  <div class="train-grid">
    <div class="stat-card"><div class="stat-val" id="st-ep">—</div><div class="stat-label">Epochs Completed</div></div>
    <div class="stat-card"><div class="stat-val" id="st-bd" style="color:#68d391">—</div><div class="stat-label">Best Val Dice</div></div>
    <div class="stat-card"><div class="stat-val" id="st-td">—</div><div class="stat-label">Latest Train Dice</div></div>
    <div class="stat-card"><div class="stat-val" id="st-vl">—</div><div class="stat-label">Latest Val Loss</div></div>
  </div>
  <div id="chartArea"><div style="font-size:12px;font-weight:600;color:#a0aec0;margin-bottom:10px;text-transform:uppercase;letter-spacing:.5px">Dice Score Progress</div>
  <canvas id="dc" height="220"></canvas></div>
  <div style="overflow-x:auto">
    <table class="epoch-table"><thead><tr><th>Ep</th><th>Train Loss</th><th>Train Dice</th><th>Val Loss</th><th>Val Dice</th><th>★</th></tr></thead>
    <tbody id="etbody"></tbody></table>
  </div>
</div></div>

<!-- ABOUT -->
<div class="section" id="s-about"><div class="container" style="max-width:700px">
  <h2 style="font-size:22px;font-weight:700;margin-bottom:20px">About ATM-Net++</h2>
  <div style="display:grid;gap:16px">
    <div class="result-card"><h3>System</h3>
      <div class="metric-row"><span>Architecture</span><strong style="color:#63b3ed">ResUNet + CBAM Attention</strong></div>
      <div class="metric-row"><span>Parameters</span><strong style="color:#63b3ed">15.7M</strong></div>
      <div class="metric-row"><span>Dataset</span><strong style="color:#63b3ed">SPIDER (210 patients, 447 scans)</strong></div>
      <div class="metric-row"><span>Training GPU</span><strong style="color:#63b3ed">NVIDIA RTX 3050 (4GB)</strong></div>
      <div class="metric-row"><span>Resolution</span><strong style="color:#63b3ed">192×192 with AMP fp16</strong></div>
    </div>
    <div class="result-card"><h3>Segmentation Classes (19)</h3>
      <div class="struct-tags">
        <span class="tag vert">Vertebra 1–8</span><span class="tag other">Sacrum</span>
        <span class="tag ivd">IVD 1–8</span><span class="tag other">Spinal Canal</span><span class="tag">Background</span>
      </div>
    </div>
    <div class="result-card"><h3>API Endpoints</h3>
      <div class="metric-row"><span>GET /</span><strong style="color:#63b3ed">Web UI</strong></div>
      <div class="metric-row"><span>POST /predict</span><strong style="color:#63b3ed">Upload file → JSON results</strong></div>
      <div class="metric-row"><span>GET /training</span><strong style="color:#63b3ed">Live training history</strong></div>
      <div class="metric-row"><span>GET /health</span><strong style="color:#63b3ed">Server & GPU status</strong></div>
    </div>
  </div>
</div></div>

<script>
let selFile=null,lastRep="";
function showTab(n,btn){
  document.querySelectorAll('.section').forEach(s=>s.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('s-'+n).classList.add('active');
  btn.classList.add('active');
  if(n==='training') loadTrain();
}
const dz=document.getElementById('dz');
dz.addEventListener('dragover',e=>{e.preventDefault();dz.classList.add('drag')});
dz.addEventListener('dragleave',()=>dz.classList.remove('drag'));
dz.addEventListener('drop',e=>{e.preventDefault();dz.classList.remove('drag');const f=e.dataTransfer.files[0];if(f)setFile(f)});
function setFile(f){
  selFile=f;
  const fi=document.getElementById('finfo');
  fi.style.display='block';fi.textContent=`✓ ${f.name}  (${(f.size/1024/1024).toFixed(2)} MB)`;
  document.getElementById('aBtn').disabled=false;
  document.getElementById('results').style.display='none';
}
async function analyze(){
  if(!selFile) return;
  document.getElementById('ov').classList.add('on');
  document.getElementById('ovTxt').textContent='Uploading & analyzing MRI...';
  const fd=new FormData(); fd.append('file',selFile);
  try{
    const r=await fetch('/predict',{method:'POST',body:fd});
    const d=await r.json();
    document.getElementById('ov').classList.remove('on');
    if(d.error){alert('Error: '+d.error);return;}
    showRes(d);
  }catch(e){document.getElementById('ov').classList.remove('on');alert('Failed: '+e.message);}
}
function showRes(d){
  document.getElementById('results').style.display='block';
  document.getElementById('itime').textContent=`Inference: ${d.inference_ms}ms | ${d.num_slices} slices`;
  document.getElementById('r-dis').textContent=d.disease;
  const sev=document.getElementById('r-sev');
  sev.textContent=d.severity;
  sev.style.color=d.severity==='None'||d.severity==='Mild'?'#68d391':d.severity==='Moderate'?'#f6ad55':'#fc8181';
  document.getElementById('r-con').textContent=d.confidence+'%';
  document.getElementById('r-pfi').textContent=d.pfirrmann_grade+'/5';
  document.getElementById('i-orig').src='data:image/png;base64,'+d.image_b64;
  document.getElementById('i-over').src='data:image/png;base64,'+d.overlay_b64;
  document.getElementById('i-mask').src='data:image/png;base64,'+d.mask_b64;
  const st=document.getElementById('stags'); st.innerHTML='';
  (d.detected_structures||[]).forEach(s=>{
    const sp=document.createElement('span');
    sp.className='tag '+(s.startsWith('Vertebra')?'vert':s.startsWith('IVD')?'ivd':'other');
    sp.textContent=s; st.appendChild(sp);
  });
  const cd=document.getElementById('cdist'); cd.innerHTML='';
  Object.entries(d.class_distribution||{}).forEach(([nm,v])=>{
    cd.innerHTML+=`<div class="metric-row"><span>${nm}</span><div style="display:flex;align-items:center;gap:10px"><div style="width:100px;background:#2d3748;border-radius:20px;height:5px"><div style="width:${Math.min(v.percent*4,100)}%;background:#3182ce;height:5px;border-radius:20px"></div></div><span style="font-size:12px;color:#63b3ed">${v.percent}%</span></div></div>`;
  });
  lastRep=d.report;
  document.getElementById('rbox').textContent=d.report;
  document.getElementById('results').scrollIntoView({behavior:'smooth'});
}
function clearAll(){selFile=null;document.getElementById('finfo').style.display='none';document.getElementById('results').style.display='none';document.getElementById('aBtn').disabled=true;document.getElementById('fi').value='';}
function dlReport(){const b=new Blob([lastRep],{type:'text/plain'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='spine_report.txt';a.click();}

async function loadTrain(){
  try{
    const r=await fetch('/training'); const d=await r.json();
    const h=d.history||[]; if(!h.length){document.getElementById('etbody').innerHTML='<tr><td colspan="6" style="text-align:center;color:#718096;padding:20px">No history yet</td></tr>';return;}
    const last=h[h.length-1]; const best=h.reduce((a,b)=>(b.vd||0)>(a.vd||0)?b:a,h[0]);
    document.getElementById('st-ep').textContent=last.ep||last.epoch||'?';
    document.getElementById('st-bd').textContent=(best.vd||0).toFixed(4);
    document.getElementById('st-td').textContent=(last.td||0).toFixed(4);
    document.getElementById('st-vl').textContent=(last.vl||0).toFixed(4);
    const tb=document.getElementById('etbody'); tb.innerHTML='';
    h.slice(-20).reverse().forEach(row=>{
      const ep=row.ep||row.epoch||0,tl=(row.tl||0).toFixed(4),td=(row.td||0).toFixed(4),vl=(row.vl||0).toFixed(4),vd=(row.vd||0).toFixed(4);
      const ib=parseFloat(vd)>=(best.vd||0)-0.0001;
      tb.innerHTML+=`<tr class="${ib?'best':''}"><td>${ep}</td><td>${tl}</td><td>${td}</td><td>${vl}</td><td><strong>${vd}</strong></td><td>${ib?'★':''}</td></tr>`;
    });
    drawDice(h);
  }catch(e){console.error(e);}
}
function drawDice(h){
  const cv=document.getElementById('dc');
  const ctx=cv.getContext('2d');
  cv.width=cv.parentElement.clientWidth-40; cv.height=220;
  const W=cv.width,H=cv.height,pad={l:45,r:20,t:20,b:35};
  ctx.clearRect(0,0,W,H);
  const eps=h.map(r=>r.ep||r.epoch||0);
  const tvd=h.map(r=>r.td||0),vvd=h.map(r=>r.vd||0);
  const maxD=Math.max(1,...tvd,...vvd);
  const minEp=eps[0]||0,maxEp=eps[eps.length-1]||1;
  function xp(e){return pad.l+(e-minEp)/(maxEp-minEp||1)*(W-pad.l-pad.r);}
  function yp(d){return H-pad.b-(d/maxD)*(H-pad.t-pad.b);}
  [0,.25,.5,.75,1].forEach(v=>{
    const y=yp(v*maxD),val=(v*maxD).toFixed(2);
    ctx.strokeStyle='#2d3748';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();
    ctx.fillStyle='#718096';ctx.font='10px sans-serif';ctx.fillText(val,2,y+3);
  });
  function line(data,col){
    ctx.strokeStyle=col;ctx.lineWidth=2.5;ctx.beginPath();
    data.forEach((d,i)=>{i===0?ctx.moveTo(xp(eps[i]),yp(d)):ctx.lineTo(xp(eps[i]),yp(d));});
    ctx.stroke();
  }
  line(tvd,'#3182ce'); line(vvd,'#68d391');
  ctx.strokeStyle='#f6ad55';ctx.lineWidth=1;ctx.setLineDash([5,4]);
  ctx.beginPath();ctx.moveTo(pad.l,yp(0.9));ctx.lineTo(W-pad.r,yp(0.9));ctx.stroke();
  ctx.setLineDash([]);ctx.fillStyle='#f6ad55';ctx.font='10px sans-serif';ctx.fillText('Target 0.90',W-90,yp(0.9)-4);
  const step=Math.max(1,Math.floor(eps.length/8));
  ctx.fillStyle='#718096';ctx.font='10px sans-serif';
  eps.forEach((e,i)=>{if(i%step===0)ctx.fillText(e,xp(e)-6,H-8);});
  ctx.fillStyle='#3182ce';ctx.fillRect(pad.l,6,12,3);ctx.fillStyle='#a0aec0';ctx.font='11px sans-serif';ctx.fillText('Train',pad.l+15,12);
  ctx.fillStyle='#68d391';ctx.fillRect(pad.l+65,6,12,3);ctx.fillStyle='#a0aec0';ctx.fillText('Val',pad.l+80,12);
}
setInterval(()=>{if(document.getElementById('s-training').classList.contains('active'))loadTrain();},60000);
</script></body></html>"""

if __name__=="__main__":
    import torch
    print("="*52)
    print("  ATM-Net++ Web Server")
    print("="*52)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    print(f"  GPU    : {gpu}")
    ck="No checkpoint"
    if CKPT and CKPT.exists():
        try:
            c=torch.load(str(CKPT),map_location="cpu")
            ck=f"epoch {c.get('epoch','?')} | best_dice={c.get('best_dice',0):.4f}"
        except: pass
    print(f"  Model  : {ck}")
    print(f"  Open   : http://localhost:5000")
    print("="*52)
    app.run(host="0.0.0.0",port=5000,debug=False,threaded=True)
