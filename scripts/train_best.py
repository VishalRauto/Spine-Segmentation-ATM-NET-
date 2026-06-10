"""
ATM-Net++ Best Training — All Expert Fixes, Fast Execution
Uses fast single-NPZ cache (127MB, loads in 2s)
Adds: weighted sampler, class-weighted Dice, boundary loss, aux head

Resume from epoch 84, Dice=0.615
Expected: Dice ~0.68-0.72 within 30 more epochs
"""

import sys, os, time, warnings, json, random, gc
warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ── CONFIG ────────────────────────────────────────────────────────────
DATA_ROOT  = Path(r"c:\project\Spine Segmentation\10159290")
IMAGES_DIR = DATA_ROOT / "images"
MASKS_DIR  = DATA_ROOT / "masks"
OVERVIEW   = DATA_ROOT / "overview.csv"
OUT_DIR    = Path(r"c:\project\Spine Segmentation\ATM-Net++\outputs\gpu_run")
CACHE_DIR  = Path(r"c:\project\Spine Segmentation\ATM-Net++\outputs\cache")
CKPT_BEST  = OUT_DIR / "best_model.pth"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE    = 192    # Keep 192 for fast cache — class weighting is the priority fix
NUM_CLASSES = 19
BATCH_SIZE  = 4
ACCUM_STEPS = 2      # effective BS=8
EPOCHS      = 200
LR          = 5e-5
WEIGHT_DECAY= 5e-4
MAX_SPP     = 20     # slices per patient for the cache
PATIENCE    = 50
SEED        = 42

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

SPIDER_TO_ATMNET = {**{i: i for i in range(1, 9)}, 100: 9,
                    **{201+i: 10+i for i in range(8)}}
CLASS_NAMES = {
    0:"background",   1:"Vert-1(L)", 2:"Vert-2",  3:"Vert-3",
    4:"Vert-4",       5:"Vert-5",    6:"Vert-6",  7:"Vert-7",
    8:"Vert-8(U)",    9:"Sacrum",
    10:"IVD-1(L)",   11:"IVD-2",    12:"IVD-3",  13:"IVD-4",
    14:"IVD-5",      15:"IVD-6",    16:"IVD-7",  17:"IVD-8(U)",
    18:"Canal",
}
VERT_CLASSES = list(range(1, 9))
IVD_CLASSES  = list(range(10, 18))
RARE_CLASSES = [7, 8, 16, 17]  # Vert-7/8, IVD-7/8

# Expert Fix: Per-class inverse-frequency weights
CLASS_WEIGHTS = torch.tensor([
    0.0,
    1.0, 1.0, 1.0, 1.5, 2.0, 3.0, 6.0, 12.0,  # Vert 1-8
    1.0,                                          # Sacrum
    6.0, 4.0, 4.0, 5.0, 6.0, 9.0, 14.0, 30.0,  # IVD 1-8
    0.0,                                          # Canal (absent)
]).float()

def remap(m):
    out = np.zeros_like(m, dtype=np.int32)
    for s, d in SPIDER_TO_ATMNET.items():
        out[m == s] = d
    return out

def fg_ratio(m): return float((m > 0).sum()) / max(m.size, 1)
def rare_ratio(m): return float(sum((m == c).sum() for c in RARE_CLASSES)) / max(m.size, 1)

# ── FAST NPZ CACHE ─────────────────────────────────────────────────────
def build_cache(pids, split):
    """Build single-file NPZ cache. Fast to load (~2s)."""
    cache_file = CACHE_DIR / f"{split}_t2_{IMG_SIZE}_best.npz"
    if cache_file.exists():
        print(f"  {split}: loading cache {cache_file.name}...", end=" ", flush=True)
        t0=time.time(); d=np.load(cache_file)
        print(f"done ({time.time()-t0:.1f}s)")
        return d["imgs"], d["msks"], d["rare"].tolist()

    print(f"  {split}: building cache for {len(pids)} patients...")
    imgs, msks, rare_flags = [], [], []
    t0 = time.time()
    for pid in pids:
        fn = f"{pid}_t2.mha"
        ip = IMAGES_DIR/fn; mp = MASKS_DIR/fn
        if not ip.exists() or not mp.exists(): continue
        try:
            iv = sitk.GetArrayFromImage(sitk.ReadImage(str(ip))).astype(np.float32)
            mv = sitk.GetArrayFromImage(sitk.ReadImage(str(mp))).astype(np.int32)
        except: continue
        n = iv.shape[0]; lo, hi = int(n*0.08), int(n*0.92)
        ranked = sorted(range(lo,hi), key=lambda s: fg_ratio(remap(mv[s])), reverse=True)[:MAX_SPP]
        for s in ranked:
            rm = remap(mv[s])
            if fg_ratio(rm) < 0.005: continue
            p1,p99 = np.percentile(iv[s],[0.5,99.5])
            img_n = np.clip((iv[s]-p1)/(p99-p1+1e-8),0,1).astype(np.float32)
            img_r = cv2.resize(img_n,(IMG_SIZE,IMG_SIZE),interpolation=cv2.INTER_LINEAR).astype(np.float16)
            msk_r = cv2.resize(rm.astype(np.float32),(IMG_SIZE,IMG_SIZE),
                               interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            imgs.append(img_r); msks.append(np.clip(msk_r,0,NUM_CLASSES-1))
            rare_flags.append(1.0 if rare_ratio(rm) > 0.0003 else 0.1)

    imgs_arr=np.stack(imgs).astype(np.float16)
    msks_arr=np.stack(msks).astype(np.uint8)
    rare_arr=np.array(rare_flags,dtype=np.float32)
    np.savez_compressed(cache_file, imgs=imgs_arr, msks=msks_arr, rare=rare_arr)
    rare_c = sum(1 for r in rare_flags if r>0.5)
    sz_mb = cache_file.stat().st_size//1024**2
    print(f"  {split}: {len(pids)} patients → {len(imgs)} slices ({rare_c} rare) | {sz_mb}MB | {time.time()-t0:.0f}s")
    return imgs_arr, msks_arr, rare_flags

# ── DATASET ────────────────────────────────────────────────────────────
class Aug:
    def __call__(self, img, msk):
        if random.random()<0.5: img=np.fliplr(img).copy(); msk=np.fliplr(msk).copy()
        if random.random()<0.6:
            a=random.uniform(-20,20)
            M=cv2.getRotationMatrix2D((IMG_SIZE//2,IMG_SIZE//2),a,1.0)
            img=cv2.warpAffine(img,M,(IMG_SIZE,IMG_SIZE),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
            mf=cv2.warpAffine(msk.astype(np.float32),M,(IMG_SIZE,IMG_SIZE),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
            msk=np.clip(mf.astype(np.int32),0,NUM_CLASSES-1)
        if random.random()<0.25:
            try:
                from scipy.ndimage import gaussian_filter, map_coordinates
                h,w=img.shape
                dx=gaussian_filter(np.random.randn(h,w),h*0.06)*(h*0.4)
                dy=gaussian_filter(np.random.randn(h,w),h*0.06)*(h*0.4)
                x,y=np.meshgrid(np.arange(w),np.arange(h))
                xi=np.clip(x+dx,0,w-1).ravel(); yi=np.clip(y+dy,0,h-1).ravel()
                img=map_coordinates(img,[yi,xi],order=1).reshape(h,w).astype(np.float32)
                mf=map_coordinates(msk.astype(float),[yi,xi],order=0).reshape(h,w)
                msk=np.clip(mf.astype(np.int32),0,NUM_CLASSES-1)
            except: pass
        gamma=random.uniform(0.65,1.5)
        img=np.clip(np.power(img.astype(np.float32)+1e-8,gamma),0,1)
        img=np.clip(img*random.uniform(0.75,1.25)+random.uniform(-0.1,0.1),0,1)
        if random.random()<0.4: img=np.clip(img+np.random.normal(0,0.015,img.shape),0,1)
        if random.random()<0.3:
            cy,cx=random.randint(0,IMG_SIZE),random.randint(0,IMG_SIZE)
            img[max(0,cy-16):min(IMG_SIZE,cy+16),max(0,cx-16):min(IMG_SIZE,cx+16)]=0
        return img.astype(np.float32), msk.astype(np.int64)

class DS(Dataset):
    def __init__(self, imgs, msks, aug=None):
        self.imgs=imgs; self.msks=msks; self.aug=aug
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        img=self.imgs[i].astype(np.float32); msk=self.msks[i].astype(np.int64)
        if self.aug: img,msk=self.aug(img,msk)
        return torch.from_numpy(img[None]).float(), torch.from_numpy(msk).long()

# ── MODEL ──────────────────────────────────────────────────────────────
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
        self.net=nn.Sequential(nn.Conv2d(ch,ch,3,1,1,bias=False),nn.BatchNorm2d(ch),nn.ReLU(True),
                               nn.Conv2d(ch,ch,3,1,1,bias=False),nn.BatchNorm2d(ch))
        self.ca=CA(ch); self.sa=SA(); self.act=nn.ReLU(True)
    def forward(self,x): return self.act(self.sa(self.ca(self.net(x)))+x)

class Enc(nn.Module):
    def __init__(self,ci,co,drop=0.0):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(ci,co,3,1,1,bias=False),nn.BatchNorm2d(co),nn.ReLU(True),
                                nn.Conv2d(co,co,3,1,1,bias=False),nn.BatchNorm2d(co),nn.ReLU(True))
        self.res=RB(co); self.drop=nn.Dropout2d(drop) if drop>0 else nn.Identity()
    def forward(self,x): return self.drop(self.res(self.conv(x)))

class ResUNet(nn.Module):
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
        sz=x.shape[2:]
        e1=self.e1(x);e2=self.e2(self.pool(e1));e3=self.e3(self.pool(e2));e4=self.e4(self.pool(e3))
        d=self.bn(self.pool(e4))
        d=self.d4(torch.cat([self.u4(d),e4],1))
        d=self.d3(torch.cat([self.u3(d),e3],1))
        o3=F.interpolate(self.ds3(d),sz,mode='bilinear',align_corners=False)
        d=self.d2(torch.cat([self.u2(d),e2],1))
        o2=F.interpolate(self.ds2(d),sz,mode='bilinear',align_corners=False)
        d=self.d1(torch.cat([self.u1(d),e1],1))
        return (self.out(d),o2,o3,self.aux(d)) if self.training else self.out(d)

# ── LOSSES ─────────────────────────────────────────────────────────────
def dice_w(logits, tgt, sm=1e-6):
    """Weighted Dice — rare classes get 30x weight."""
    B,C,H,W=logits.shape
    soft=F.softmax(logits,1)
    oh  =F.one_hot(tgt.clamp(0,C-1),C).permute(0,3,1,2).float()
    p=soft[:,1:].reshape(B,C-1,-1); t=oh[:,1:].reshape(B,C-1,-1)
    inter=(p*t).sum(-1); union=p.sum(-1)+t.sum(-1)
    mask=(t.sum(-1)>0).float()
    dice=(2*inter+sm)/(union+sm)
    w=CLASS_WEIGHTS[1:].to(logits.device).view(1,C-1)
    return 1.0-(dice*mask*w).sum()/(mask*w).sum().clamp(min=1)

def boundary_l(logits, tgt):
    """Boundary loss — penalize near-boundary misclassification."""
    soft=F.softmax(logits,1)
    oh=F.one_hot(tgt.clamp(0,NUM_CLASSES-1),NUM_CLASSES).permute(0,3,1,2).float()
    pooled=F.max_pool2d(oh[:,1:],3,stride=1,padding=1)
    boundary=(pooled-oh[:,1:]).clamp(0,1)
    w=CLASS_WEIGHTS[1:].to(logits.device).view(1,-1,1,1)
    return (boundary*(1-soft[:,1:])*w).sum()/((boundary*w).sum()+1e-6)

def focal_l(logits, tgt, gamma=2.0):
    ce=F.cross_entropy(logits,tgt.clamp(0,NUM_CLASSES-1),reduction='none')
    return ((1-torch.exp(-ce))**gamma*ce).mean()

def compound(logits, tgt):
    tc=tgt.clamp(0,NUM_CLASSES-1)
    return (F.cross_entropy(logits,tc,label_smoothing=0.03)
            + dice_w(logits,tc)
            + 0.3*focal_l(logits,tc)
            + 0.15*boundary_l(logits,tc))

def total_loss(outs, tgt):
    o1,o2,o3,aux=outs
    main=compound(o1,tgt)+0.3*compound(o2,tgt)+0.15*compound(o3,tgt)
    # Aux head: extra weight on rare pixels
    tc=tgt.clamp(0,NUM_CLASSES-1)
    rare_mask=sum((tc==c).float() for c in RARE_CLASSES).clamp(0,1)
    aux_ce=F.cross_entropy(aux,tc,reduction='none')
    return main + 0.3*(aux_ce*(1+4*rare_mask)).mean()

# ── METRICS ────────────────────────────────────────────────────────────
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

# ── SPLITS ─────────────────────────────────────────────────────────────
def get_splits():
    df=pd.read_csv(OVERVIEW); tr,va=[],[]
    seen=set()
    for name in df["new_file_name"].tolist():
        if not name.endswith("_t2") or "SPACE" in name: continue
        pid=name.replace("_t2","")
        if pid in seen: continue
        seen.add(pid)
        sub=df.loc[df["new_file_name"]==name,"subset"].values
        (va if len(sub) and sub[0]=="validation" else tr).append(pid)
    return tr, va

# ── MAIN ───────────────────────────────────────────────────────────────
def main():
    print("="*65)
    print("  ATM-Net++ Best Training — All Expert Fixes Applied")
    print("  Weighted sampler + class weights + boundary loss + aux head")
    print("="*65)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  GPU    : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    tr_pids, va_pids = get_splits()
    print(f"  Data   : {len(tr_pids)} train | {len(va_pids)} val")
    print(f"  Config : {EPOCHS} epochs  BS={BATCH_SIZE}x{ACCUM_STEPS}=eff{BATCH_SIZE*ACCUM_STEPS}"
          f"  LR={LR}  {IMG_SIZE}x{IMG_SIZE}\n")

    print("  Loading data...")
    ti,tm,tr_rare = build_cache(tr_pids, "train_best")
    vi,vm,va_rare = build_cache(va_pids, "val_best")
    print(f"  RAM: ~{(ti.nbytes+tm.nbytes+vi.nbytes+vm.nbytes)//1024**2} MB")

    rare_cnt = sum(1 for r in tr_rare if r > 0.5)
    print(f"  Rare slices: {rare_cnt}/{len(tr_rare)} (10x oversampled by WeightedSampler)\n")

    aug=Aug()
    tr_ds=DS(ti,tm,aug); va_ds=DS(vi,vm)

    # Weighted sampler: rare classes (Vert-7/8, IVD-7/8) get 10x sampling
    sampler=WeightedRandomSampler(torch.tensor(tr_rare), len(tr_rare), replacement=True)
    tr_dl=DataLoader(tr_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    va_dl=DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,   num_workers=0)

    model=ResUNet(b=32,nc=NUM_CLASSES,drop=0.25).to(device)
    n_p=sum(p.numel() for p in model.parameters())
    print(f"  Model  : ResUNet+CBAM+AuxHead  {n_p/1e6:.2f}M params")
    print(f"  Batches: {len(tr_dl)} train | {len(va_dl)} val/epoch")

    start_epoch=1; best=0.0
    if CKPT_BEST.exists():
        ckpt=torch.load(str(CKPT_BEST),map_location=device)
        missing,_=model.load_state_dict(ckpt["model_state_dict"],strict=False)
        best=ckpt.get("best_dice",0.0)
        start_epoch=ckpt.get("epoch",84)+1
        n_new=len(missing) if missing else 0
        print(f"\n  Loaded : epoch {ckpt.get('epoch')} | best={best:.4f} | {n_new} new keys (aux)")
        print(f"  Resume : from epoch {start_epoch}\n")

    optim=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(EPOCHS-start_epoch+1,1), eta_min=1e-7)

    no_imp=0; hist=[]; t0tot=time.time(); ep=start_epoch-1

    print(f"  {'Ep':>4}  {'TrLoss':>8}  {'TrDice':>8}  {'VaLoss':>8}  "
          f"{'VaDice':>8}  {'Best':>8}  {'Gap':>6}  {'Sec':>5}")
    print("  "+"─"*65)

    for ep in range(start_epoch, EPOCHS+1):
        torch.cuda.empty_cache()
        model.train(); tl_b,td_b=[],[]; t0=time.time()
        optim.zero_grad(set_to_none=True)

        for i,(imgs,msks) in enumerate(tr_dl):
            imgs,msks=imgs.to(device),msks.to(device)
            outs=model(imgs)
            loss=total_loss(outs,msks)/ACCUM_STEPS
            loss.backward()
            if (i+1)%ACCUM_STEPS==0 or (i+1)==len(tr_dl):
                nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optim.step(); optim.zero_grad(set_to_none=True)
            tl_b.append(loss.item()*ACCUM_STEPS)
            with torch.no_grad():
                _,d=batch_dice(outs[0],msks); td_b.append(d)

        sched.step()
        tl=float(np.mean(tl_b)); td=float(np.mean(td_b)); ep_s=time.time()-t0

        model.eval(); vl_b=[]; Dc=defaultdict(list)
        with torch.no_grad():
            for imgs,msks in va_dl:
                imgs,msks=imgs.to(device),msks.to(device)
                out=model(imgs)
                vl_b.append(compound(out,msks).item())
                D,_=batch_dice(out,msks)
                for c,v in D.items(): Dc[c].extend(v)

        vl=float(np.mean(vl_b))
        all_v=[v for vs in Dc.values() for v in vs]
        vd=float(np.mean(all_v)) if all_v else 0.0
        lr_n=float(optim.param_groups[0]['lr']); gap=td-vd

        if vd>best:
            best=vd; no_imp=0
            pc={CLASS_NAMES[c]:float(np.mean(v)) for c,v in Dc.items() if v}
            torch.save({"epoch":ep,"model_state_dict":model.state_dict(),
                        "best_dice":best,"per_class_dice":pc,
                        "cfg":{"sz":IMG_SIZE,"nc":NUM_CLASSES}},CKPT_BEST)
        else: no_imp+=1

        if ep%10==0:
            torch.save({"epoch":ep,"model_state_dict":model.state_dict(),"best_dice":best},
                       OUT_DIR/"last_model.pth")

        flag=" ★" if vd==best else ""
        print(f"  {ep:>4}  {tl:>8.4f}  {td:>8.4f}  {vl:>8.4f}  "
              f"{vd:>8.4f}  {best:>8.4f}  {gap:>+6.3f}  {ep_s:>4.0f}s{flag}")
        hist.append({"ep":ep,"tl":tl,"td":td,"vl":vl,"vd":vd,"gap":gap,"lr":lr_n})

        if no_imp>=PATIENCE:
            print(f"\n  Early stop at ep {ep}"); break
        gc.collect()

    ttot=time.time()-t0tot
    print("  "+"─"*65)
    print(f"\n  Done in {ttot/60:.1f}min  |  Best Dice: {best:.4f}")
    with open(OUT_DIR/"history.json","w") as f: json.dump(hist,f,indent=2)

    # ── FINAL EVALUATION WITH TTA ────────────────────────────────────
    print("\n"+"="*65)
    print("  FINAL EVALUATION (best checkpoint + TTA)")
    print("="*65)
    ckpt=torch.load(str(CKPT_BEST),map_location=device)
    model.load_state_dict(ckpt["model_state_dict"],strict=False); model.eval()
    print(f"  Epoch: {ckpt['epoch']} | dice={ckpt['best_dice']:.4f}\n")

    aD=defaultdict(list); aI=defaultdict(list); n_sl=0; sm=1e-6
    with torch.no_grad():
        for imgs,msks in va_dl:
            imgs,msks=imgs.to(device),msks.to(device)
            p1=F.softmax(model(imgs),1)
            p2=F.softmax(model(torch.flip(imgs,[-1])),1)
            probs=(p1+torch.flip(p2,[-1]))/2
            pnp=probs.argmax(1).cpu().numpy(); gnp=msks.cpu().numpy()
            for b in range(pnp.shape[0]):
                n_sl+=1
                for c in range(1,NUM_CLASSES):
                    p=(pnp[b]==c).astype(float).ravel(); t=(gnp[b]==c).astype(float).ravel()
                    if t.sum()==0 and p.sum()==0: continue
                    tp=(p*t).sum(); fp=(p*(1-t)).sum(); fn=((1-p)*t).sum()
                    aD[c].append((2*tp+sm)/(2*tp+fp+fn+sm))
                    aI[c].append((tp+sm)/(tp+fp+fn+sm))

    print(f"  {'Class':<22} {'Dice':>8}  {'IoU':>8}  {'N':>6}")
    print("  "+"─"*44)
    vd_all=[]; vi_all=[]; vert_d=[]; ivd_d=[]; res={}
    for c in range(1,NUM_CLASSES):
        if not aD[c]: continue
        d=float(np.mean(aD[c])); io=float(np.mean(aI[c])); n=len(aD[c])
        nm=CLASS_NAMES[c]; res[nm]={"dice":d,"iou":io,"n":n}
        vd_all.append(d); vi_all.append(io)
        if c in VERT_CLASSES: vert_d.append(d)
        if c in IVD_CLASSES:  ivd_d.append(d)
        tag="  ★" if d>=0.90 else "  ✓" if d>=0.80 else "  ·" if d>=0.70 else "  +" if d>=0.60 else ""
        print(f"  {nm:<22} {d:>8.4f}  {io:>8.4f}  {n:>6}{tag}")

    md=float(np.mean(vd_all)) if vd_all else 0.0
    mi=float(np.mean(vi_all)) if vi_all else 0.0
    vdm=float(np.mean(vert_d)) if vert_d else 0.0
    idm=float(np.mean(ivd_d))  if ivd_d  else 0.0
    print("  "+"─"*44)
    print(f"  {'MEAN FOREGROUND':<22} {md:>8.4f}  {mi:>8.4f}")

    print(f"""
  ╔════════════════════════════════════════════════════╗
  ║  Vertebrae Dice  : {vdm:.4f}                       ║
  ║  IVD Dice        : {idm:.4f}                       ║
  ║  Mean Dice (TTA) : {md:.4f}                       ║
  ║  Mean IoU  (TTA) : {mi:.4f}                       ║
  ║  Val slices      : {n_sl}                         ║
  ╚════════════════════════════════════════════════════╝
""")
    final={"mean_dice":md,"mean_iou":mi,"vertebrae_dice":vdm,"ivd_dice":idm,
           "total_val_slices":n_sl,"per_class":res}
    with open(OUT_DIR/"evaluation_results.json","w") as f: json.dump(final,f,indent=2)
    print(f"  Results → {OUT_DIR/'evaluation_results.json'}")
    print("="*65)

if __name__=="__main__":
    main()
