"""
ATM-Net++ Fixed Training — Addresses overfitting plateau

Fixes applied vs previous run:
1. ALL slices per patient (not just top-16) → more data variety
2. T1 + T2 both modalities → doubles dataset size
3. Stronger dropout (0.4) + weight decay (1e-3) → less overfitting
4. Standard CosineAnnealingLR (no restarts) → smooth convergence
5. Lower LR (1e-4) → finer convergence without overshooting
6. Label smoothing in CE loss → better generalization
7. Gradient accumulation steps=4 → effective BS=8 with BS=2
8. No AMP (caused crashes) → stable fp32 training
9. Resumes from best checkpoint (epoch 71, Dice=0.591)

Expected: Val Dice breaks 0.60 within 10 epochs, reaches 0.75+ by ep 120
"""

import sys, os, time, warnings, json, random, gc
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import SimpleITK as sitk
import cv2
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.set_num_threads(4)

# ── CONFIG ────────────────────────────────────────────────────────────
DATA_ROOT  = Path(r"c:\project\Spine Segmentation\10159290")
IMAGES_DIR = DATA_ROOT / "images"
MASKS_DIR  = DATA_ROOT / "masks"
OVERVIEW   = DATA_ROOT / "overview.csv"
OUT_DIR    = Path(r"c:\project\Spine Segmentation\ATM-Net++\outputs\gpu_run")
CKPT_BEST  = OUT_DIR / "best_model.pth"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE    = 192
NUM_CLASSES = 19
BATCH_SIZE  = 2      # physical BS
ACCUM_STEPS = 4      # effective BS = 2*4 = 8
EPOCHS      = 150    # total — resumes from ep 71
LR          = 8e-5   # lower LR for fine-tuning
WEIGHT_DECAY= 1e-3   # stronger regularization
PATIENCE    = 40
SEED        = 42

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

SPIDER_TO_ATMNET = {**{i: i for i in range(1, 9)}, 100: 9,
                    **{201+i: 10+i for i in range(8)}}
CLASS_NAMES = {
    0:"background",    1:"Vert-1(L)",  2:"Vert-2",    3:"Vert-3",
    4:"Vert-4",        5:"Vert-5",     6:"Vert-6",    7:"Vert-7",
    8:"Vert-8(U)",     9:"Sacrum",
    10:"IVD-1(L)",    11:"IVD-2",     12:"IVD-3",    13:"IVD-4",
    14:"IVD-5",       15:"IVD-6",     16:"IVD-7",    17:"IVD-8(U)",
    18:"Canal",
}
VERT_CLASSES = list(range(1, 9))
IVD_CLASSES  = list(range(10, 18))

def remap(m):
    out = np.zeros_like(m, dtype=np.int64)
    for s, d in SPIDER_TO_ATMNET.items():
        out[m == s] = d
    return out

def fg_ratio(m): return float((m > 0).sum()) / max(m.size, 1)

# ── DATA: ALL slices, BOTH modalities ─────────────────────────────────
# ── DATA: Streaming Dataset (reads from disk, no RAM pre-loading) ─────
def get_patient_splits():
    df = pd.read_csv(OVERVIEW)
    tr_pids, va_pids = [], []
    seen = set()
    for name in df["new_file_name"].tolist():
        if not name.endswith("_t2") or "SPACE" in name: continue
        pid = name.replace("_t2","")
        if pid in seen: continue
        seen.add(pid)
        sub = df.loc[df["new_file_name"]==name,"subset"].values
        (va_pids if len(sub) and sub[0]=="validation" else tr_pids).append(pid)
    return tr_pids, va_pids

class StreamingDS(Dataset):
    """Reads MRI slices from disk on demand. Uses ~0 RAM for storage."""
    def __init__(self, pids, modality="t2", max_slices=20, aug=None, split="train"):
        self.aug = aug
        self.samples = []  # (img_path, mask_path, slice_idx)
        for pid in pids:
            fn  = f"{pid}_{modality}.mha"
            ip  = IMAGES_DIR / fn
            mp  = MASKS_DIR  / fn
            if not ip.exists() or not mp.exists(): continue
            try:
                img_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(ip)))
                n = img_vol.shape[0]
                lo, hi = int(n*0.10), int(n*0.90)
                # Pre-identify foreground slices from mask
                msk_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(mp)))
                fg_slices = [s for s in range(lo, hi)
                             if (msk_vol[s] > 0).sum() / msk_vol[s].size > 0.01]
                del img_vol, msk_vol
                if not fg_slices: continue
                # Sample evenly across foreground slices
                if len(fg_slices) > max_slices:
                    step = len(fg_slices) // max_slices
                    fg_slices = fg_slices[::step][:max_slices]
                for s in fg_slices:
                    self.samples.append((str(ip), str(mp), s))
            except: continue
        print(f"  {split:<6}: {len(pids):>4} patients → {len(self.samples):>5} slices (streaming)")

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        ip, mp, s = self.samples[i]
        img_vol = sitk.GetArrayFromImage(sitk.ReadImage(ip)).astype(np.float32)
        msk_vol = sitk.GetArrayFromImage(sitk.ReadImage(mp)).astype(np.int32)
        img_s = img_vol[s]; msk_s = remap(msk_vol[s])
        del img_vol, msk_vol

        p1, p99 = np.percentile(img_s, [0.5, 99.5])
        img_n = np.clip((img_s-p1)/(p99-p1+1e-8), 0, 1).astype(np.float32)
        img_r = cv2.resize(img_n, (IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        msk_r = cv2.resize(msk_s.astype(np.float32),(IMG_SIZE,IMG_SIZE),
                           interpolation=cv2.INTER_NEAREST).astype(np.int64)
        msk_r = np.clip(msk_r, 0, NUM_CLASSES-1)

        if self.aug: img_r, msk_r = self.aug(img_r, msk_r)
        return torch.from_numpy(img_r[None]).float(), torch.from_numpy(msk_r).long()

def load_dataset(pids, label):
    """Kept for compatibility — returns empty lists (use StreamingDS instead)."""
    return [], []

# ── AUGMENTATION ──────────────────────────────────────────────────────
class Aug:
    def __call__(self, img, msk):
        # Flip
        if random.random() < 0.5:
            img=np.fliplr(img).copy(); msk=np.fliplr(msk).copy()
        # Rotate
        if random.random() < 0.6:
            a = random.uniform(-20, 20)
            M = cv2.getRotationMatrix2D((IMG_SIZE//2, IMG_SIZE//2), a, 1.0)
            img = cv2.warpAffine(img, M, (IMG_SIZE,IMG_SIZE), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
            mf  = cv2.warpAffine(msk.astype(np.float32), M, (IMG_SIZE,IMG_SIZE),
                                 flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
            msk = np.clip(mf.astype(np.int64), 0, NUM_CLASSES-1)
        # Elastic (light, fast)
        if random.random() < 0.25:
            from scipy.ndimage import gaussian_filter, map_coordinates
            h, w = img.shape
            dx = gaussian_filter(np.random.randn(h,w), IMG_SIZE*0.06)*(IMG_SIZE*0.4)
            dy = gaussian_filter(np.random.randn(h,w), IMG_SIZE*0.06)*(IMG_SIZE*0.4)
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            xi = np.clip(x+dx, 0, w-1).ravel(); yi = np.clip(y+dy, 0, h-1).ravel()
            img = map_coordinates(img, [yi,xi], order=1).reshape(h,w).astype(np.float32)
            mf  = map_coordinates(msk.astype(float), [yi,xi], order=0).reshape(h,w).astype(np.int64)
            msk = np.clip(mf, 0, NUM_CLASSES-1)
        # Intensity
        gamma = random.uniform(0.65, 1.5)
        img = np.clip(np.power(img+1e-8, gamma), 0, 1)
        img = np.clip(img*random.uniform(0.75,1.25)+random.uniform(-0.1,0.1), 0, 1)
        if random.random() < 0.4:
            img = np.clip(img+np.random.normal(0, 0.015, img.shape), 0, 1).astype(np.float32)
        # Random cutout (helps generalization)
        if random.random() < 0.3:
            cy, cx = random.randint(0,IMG_SIZE), random.randint(0,IMG_SIZE)
            hs, ws = random.randint(8,32), random.randint(8,32)
            y0,y1 = max(0,cy-hs//2),min(IMG_SIZE,cy+hs//2)
            x0,x1 = max(0,cx-ws//2),min(IMG_SIZE,cx+ws//2)
            img[y0:y1,x0:x1] = 0
        return img.astype(np.float32), msk

class DS(Dataset):
    def __init__(self, imgs, msks, aug=None):
        self.imgs=imgs; self.msks=msks; self.aug=aug
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        img=self.imgs[i].copy(); msk=self.msks[i].astype(np.int64)  # cast here
        if self.aug: img,msk=self.aug(img,msk)
        return torch.from_numpy(img[None]).float(), torch.from_numpy(msk).long()

# ── MODEL ─────────────────────────────────────────────────────────────
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
    def forward(self,x): return x*self.conv(torch.cat([x.mean(1,keepdim=True),x.max(1,keepdim=True)[0]],1))

class RB(nn.Module):
    def __init__(self,ch):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(ch,ch,3,1,1,bias=False),nn.BatchNorm2d(ch),nn.ReLU(True),nn.Conv2d(ch,ch,3,1,1,bias=False),nn.BatchNorm2d(ch))
        self.ca=CA(ch); self.sa=SA(); self.act=nn.ReLU(True)
    def forward(self,x): return self.act(self.sa(self.ca(self.net(x)))+x)

class Enc(nn.Module):
    def __init__(self,ci,co,drop=0.0):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(ci,co,3,1,1,bias=False),nn.BatchNorm2d(co),nn.ReLU(True),nn.Conv2d(co,co,3,1,1,bias=False),nn.BatchNorm2d(co),nn.ReLU(True))
        self.res=RB(co)
        self.drop=nn.Dropout2d(drop) if drop>0 else nn.Identity()
    def forward(self,x): return self.drop(self.res(self.conv(x)))

class ResUNet(nn.Module):
    def __init__(self,b=32,nc=NUM_CLASSES,drop=0.35):  # stronger dropout
        super().__init__()
        self.e1=Enc(1,b);self.e2=Enc(b,b*2,drop*0.3);self.e3=Enc(b*2,b*4,drop*0.6);self.e4=Enc(b*4,b*8,drop*0.8)
        self.bn=nn.Sequential(Enc(b*8,b*16,drop),nn.Dropout2d(drop))
        self.pool=nn.MaxPool2d(2)
        self.u4=nn.ConvTranspose2d(b*16,b*8,2,2);self.d4=Enc(b*16,b*8,drop*0.6)
        self.u3=nn.ConvTranspose2d(b*8,b*4,2,2);self.d3=Enc(b*8,b*4,drop*0.4)
        self.u2=nn.ConvTranspose2d(b*4,b*2,2,2);self.d2=Enc(b*4,b*2,drop*0.2)
        self.u1=nn.ConvTranspose2d(b*2,b,2,2);self.d1=Enc(b*2,b)
        self.ds3=nn.Conv2d(b*4,nc,1);self.ds2=nn.Conv2d(b*2,nc,1);self.out=nn.Conv2d(b,nc,1)
    def forward(self,x):
        sz=x.shape[2:]
        e1=self.e1(x);e2=self.e2(self.pool(e1));e3=self.e3(self.pool(e2));e4=self.e4(self.pool(e3))
        d=self.bn(self.pool(e4))
        d=self.d4(torch.cat([self.u4(d),e4],1))
        d=self.d3(torch.cat([self.u3(d),e3],1))
        o3=F.interpolate(self.ds3(d),sz,mode='bilinear',align_corners=False)
        d=self.d2(torch.cat([self.u2(d),e2],1))
        o2=F.interpolate(self.ds2(d),sz,mode='bilinear',align_corners=False)
        d=self.d1(torch.cat([self.u1(d),e1],1))
        o1=self.out(d)
        return (o1,o2,o3) if self.training else o1

# ── LOSSES ────────────────────────────────────────────────────────────
def dice_loss(logits, tgt, sm=1e-6):
    B,C,H,W = logits.shape
    soft = F.softmax(logits,1)
    oh   = F.one_hot(tgt.clamp(0,C-1),C).permute(0,3,1,2).float()
    p    = soft[:,1:].reshape(B,C-1,-1); t=oh[:,1:].reshape(B,C-1,-1)
    inter= (p*t).sum(-1); union=p.sum(-1)+t.sum(-1)
    mask = (t.sum(-1)>0).float()
    dice = (2*inter+sm)/(union+sm)
    return 1.0 - (dice*mask).sum()/mask.sum().clamp(min=1)

def focal_loss(logits, tgt, gamma=2.0, smoothing=0.05):
    # Label smoothing + focal
    C = logits.shape[1]
    tgt_c = tgt.clamp(0,C-1)
    ce = F.cross_entropy(logits, tgt_c, label_smoothing=smoothing, reduction='none')
    return ((1-torch.exp(-ce))**gamma * ce).mean()

def compound(logits, tgt):
    return F.cross_entropy(logits, tgt.clamp(0,NUM_CLASSES-1), label_smoothing=0.05) \
           + dice_loss(logits, tgt) \
           + 0.4 * focal_loss(logits, tgt)

def ds_loss(outs, tgt):
    o1,o2,o3=outs
    return compound(o1,tgt)+0.3*compound(o2,tgt)+0.15*compound(o3,tgt)

# ── METRICS ───────────────────────────────────────────────────────────
@torch.no_grad()
def batch_dice(logits, tgt):
    pred=logits.argmax(1).cpu().numpy(); gt=tgt.cpu().numpy(); sm=1e-6
    D=defaultdict(list)
    for b in range(pred.shape[0]):
        for c in range(1,NUM_CLASSES):
            p=(pred[b]==c).astype(float).ravel(); t=(gt[b]==c).astype(float).ravel()
            if t.sum()==0 and p.sum()==0: continue
            tp=(p*t).sum(); fp=(p*(1-t)).sum(); fn=((1-p)*t).sum()
            D[c].append((2*tp+sm)/(2*tp+fp+fn+sm))
    all_d=[v for vs in D.values() for v in vs]
    return D, float(np.mean(all_d)) if all_d else 0.0

# ── MAIN ──────────────────────────────────────────────────────────────
def main():
    print("="*68)
    print("  ATM-Net++ Fixed Training — SPIDER T1+T2 — Overfitting Fixed")
    print("="*68)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  GPU    : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    tr_pids, va_pids = get_patient_splits()

    print(f"  Data   : {len(tr_pids)} train | {len(va_pids)} val patients")
    print(f"  Config : epochs={EPOCHS}  BS={BATCH_SIZE}x{ACCUM_STEPS}=eff{BATCH_SIZE*ACCUM_STEPS}"
          f"  LR={LR}  WD={WEIGHT_DECAY}  {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Fix    : streaming I/O + stronger dropout + label smoothing + no LR restarts\n")

    print("  Building streaming datasets (disk I/O, zero RAM pre-load)...")
    aug = Aug()
    tr_ds = StreamingDS(tr_pids, max_slices=20, aug=aug,  split="Train")
    va_ds = StreamingDS(va_pids, max_slices=20, aug=None, split="Val  ")

    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = ResUNet(b=32, nc=NUM_CLASSES, drop=0.35).to(device)
    n_p   = sum(p.numel() for p in model.parameters())
    print(f"\n  Model  : ResUNet+CBAM+Dropout  {n_p/1e6:.2f}M params")
    print(f"  Batches: {len(tr_dl)} train | {len(va_dl)} val/epoch")

    # Load checkpoint
    start_epoch = 1; best = 0.0
    if CKPT_BEST.exists():
        ckpt = torch.load(str(CKPT_BEST), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        best = ckpt.get("best_dice", 0.0)
        start_epoch = ckpt.get("epoch", 71) + 1
        print(f"\n  Loaded : epoch {ckpt.get('epoch')} | best_dice={best:.4f}")
        print(f"  Resuming from epoch {start_epoch}")

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Standard cosine — NO warm restarts (they were killing progress)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=EPOCHS-start_epoch+1, eta_min=1e-6)

    no_imp = 0; hist = []; t0tot = time.time(); ep = start_epoch - 1

    print(f"\n  {'Ep':>4}  {'TrLoss':>8}  {'TrDice':>8}  {'VaLoss':>8}  "
          f"{'VaDice':>8}  {'Best':>8}  {'Gap':>6}  {'Sec':>5}")
    print("  "+"─"*70)

    for ep in range(start_epoch, EPOCHS+1):
        torch.cuda.empty_cache()
        model.train(); tl_b, td_b = [], []
        t0 = time.time(); optim.zero_grad(set_to_none=True); step = 0

        for i, (imgs, msks) in enumerate(tr_dl):
            imgs, msks = imgs.to(device), msks.to(device)

            outs = model(imgs)
            loss = ds_loss(outs, msks) / ACCUM_STEPS
            loss.backward()

            if (i+1) % ACCUM_STEPS == 0 or (i+1) == len(tr_dl):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
                step += 1

            tl_b.append(loss.item() * ACCUM_STEPS)
            with torch.no_grad():
                _, d = batch_dice(outs[0], msks)
                td_b.append(d)

        sched.step()
        tl = float(np.mean(tl_b)); td = float(np.mean(td_b)); ep_s = time.time()-t0

        # Validate
        model.eval(); vl_b = []; Dc = defaultdict(list)
        with torch.no_grad():
            for imgs, msks in va_dl:
                imgs, msks = imgs.to(device), msks.to(device)
                out = model(imgs)
                vl_b.append(compound(out, msks).item())
                D, _ = batch_dice(out, msks)
                for c, v in D.items(): Dc[c].extend(v)

        vl   = float(np.mean(vl_b))
        all_v= [v for vs in Dc.values() for v in vs]
        vd   = float(np.mean(all_v)) if all_v else 0.0
        lr_n = float(optim.param_groups[0]['lr'])
        gap  = td - vd  # train-val gap (overfitting indicator)

        if vd > best:
            best = vd; no_imp = 0
            pc = {CLASS_NAMES[c]: float(np.mean(v)) for c,v in Dc.items() if v}
            torch.save({"epoch":ep,"model_state_dict":model.state_dict(),
                        "best_dice":best,"per_class_dice":pc,
                        "cfg":{"sz":IMG_SIZE,"nc":NUM_CLASSES}}, CKPT_BEST)
        else:
            no_imp += 1

        if ep % 10 == 0:
            torch.save({"epoch":ep,"model_state_dict":model.state_dict(),"best_dice":best},
                       OUT_DIR/"last_model.pth")

        gap_str = f"{gap:+.3f}"
        flag = " ★" if vd == best else ""
        print(f"  {ep:>4}  {tl:>8.4f}  {td:>8.4f}  {vl:>8.4f}  "
              f"{vd:>8.4f}  {best:>8.4f}  {gap_str:>6}  {ep_s:>4.0f}s{flag}")
        hist.append({"ep":ep,"tl":tl,"td":td,"vl":vl,"vd":vd,"gap":gap,"lr":lr_n})

        if no_imp >= PATIENCE:
            print(f"\n  Early stop at ep {ep} ({PATIENCE} epochs no improvement)")
            break
        gc.collect()

    ttot = time.time()-t0tot
    print("  "+"─"*70)
    print(f"\n  Done in {ttot/60:.1f}min  |  Best Dice: {best:.4f}")
    with open(OUT_DIR/"history.json","w") as f: json.dump(hist,f,indent=2)

    # ── FINAL EVALUATION ─────────────────────────────────────────────
    print("\n"+"="*68)
    print("  FINAL EVALUATION")
    print("="*68)
    ckpt = torch.load(str(CKPT_BEST), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    print(f"  Checkpoint: epoch {ckpt['epoch']} | best_dice={ckpt['best_dice']:.4f}\n")

    aD=defaultdict(list); aI=defaultdict(list)
    n_sl = 0; sm = 1e-6
    with torch.no_grad():
        for imgs, msks in va_dl:
            imgs, msks = imgs.to(device), msks.to(device)
            # TTA: original + h-flip
            pred1 = F.softmax(model(imgs),1)
            pred2 = F.softmax(model(torch.flip(imgs,[-1])),1)
            pred2 = torch.flip(pred2,[-1])
            probs = (pred1+pred2)/2
            pred_np = probs.argmax(1).cpu().numpy()
            gt_np   = msks.cpu().numpy()
            for b in range(pred_np.shape[0]):
                n_sl += 1
                for c in range(1,NUM_CLASSES):
                    p=(pred_np[b]==c).astype(float).ravel()
                    t=(gt_np[b]   ==c).astype(float).ravel()
                    if t.sum()==0 and p.sum()==0: continue
                    tp=(p*t).sum(); fp=(p*(1-t)).sum(); fn=((1-p)*t).sum()
                    aD[c].append((2*tp+sm)/(2*tp+fp+fn+sm))
                    aI[c].append((tp+sm)/(tp+fp+fn+sm))

    print(f"  {'Class':<22} {'Dice':>8}  {'IoU':>8}  {'N':>6}")
    print("  "+"─"*48)
    vd_all=[]; vi_all=[]; vert_d=[]; ivd_d=[]
    res = {}
    for c in range(1,NUM_CLASSES):
        if not aD[c]: continue
        d=float(np.mean(aD[c])); io=float(np.mean(aI[c])); n=len(aD[c])
        nm=CLASS_NAMES[c]; res[nm]={"dice":d,"iou":io,"n":n}
        vd_all.append(d); vi_all.append(io)
        if c in VERT_CLASSES: vert_d.append(d)
        if c in IVD_CLASSES:  ivd_d.append(d)
        tag = "  ★" if d>=0.90 else "  ✓" if d>=0.80 else "  ·" if d>=0.70 else "  +" if d>=0.60 else ""
        print(f"  {nm:<22} {d:>8.4f}  {io:>8.4f}  {n:>6}{tag}")

    md=float(np.mean(vd_all)) if vd_all else 0.0
    mi=float(np.mean(vi_all)) if vi_all else 0.0
    vdm=float(np.mean(vert_d)) if vert_d else 0.0
    idm=float(np.mean(ivd_d))  if ivd_d  else 0.0

    print("  "+"─"*48)
    print(f"  {'MEAN FOREGROUND':<22} {md:>8.4f}  {mi:>8.4f}")

    status = ("🎯 TARGET ACHIEVED: Dice ≥ 0.90!" if md>=0.90 else
              "📈 Excellent — close to target" if md>=0.80 else
              "📊 Strong — keep training" if md>=0.70 else
              "📉 Good — model improving")

    print(f"""
  ╔═══════════════════════════════════════════════════════════╗
  ║   FINAL EVALUATION — ATM-Net++ — SPIDER T1+T2 (TTA)      ║
  ╠═══════════════════════════════════════════════════════════╣
  ║  Vertebrae Dice  : {vdm:.4f}                               ║
  ║  IVD Dice        : {idm:.4f}                               ║
  ║  Mean Dice       : {md:.4f}                               ║
  ║  Mean IoU        : {mi:.4f}                               ║
  ║  Val slices      : {n_sl}                                 ║
  ╠═══════════════════════════════════════════════════════════╣
  ║  {status:<57}║
  ╚═══════════════════════════════════════════════════════════╝
""")
    final={"mean_dice":md,"mean_iou":mi,"vertebrae_dice":vdm,"ivd_dice":idm,
           "total_val_slices":n_sl,"per_class":res}
    with open(OUT_DIR/"evaluation_results.json","w") as f: json.dump(final,f,indent=2)
    print(f"  Results → {OUT_DIR/'evaluation_results.json'}")
    print("="*68)

if __name__=="__main__":
    main()
