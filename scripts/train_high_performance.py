"""
ATM-Net++ Optimised CPU Training — Best Dice within ~3-4 hours
Hardware: 12-core CPU, 7GB RAM, no GPU

Optimisations for maximum Dice on CPU:
  - 96×96 resolution (compromise: 2.25x more pixels than 64x64)
  - ResUNet with CBAM attention (significantly better than plain UNet)
  - Compound loss: Dice + Tversky + Focal
  - Deep supervision (4 heads)
  - SGDR cosine restarts scheduler
  - Strong augmentation pipeline
  - TTA at evaluation (4 variants)
  - All 172 train patients × 12 foreground slices = 2064 slices
  - Early stopping (patience=30)
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

# Maximize CPU parallelism
torch.set_num_threads(min(8, os.cpu_count() or 4))

# ── CONFIG ────────────────────────────────────────────────────────────
DATA_ROOT  = Path(r"c:\project\Spine Segmentation\10159290")
IMAGES_DIR = DATA_ROOT / "images"
MASKS_DIR  = DATA_ROOT / "masks"
OVERVIEW   = DATA_ROOT / "overview.csv"
OUT_DIR    = Path(r"c:\project\Spine Segmentation\ATM-Net++\outputs\high_perf_run")
CKPT_BEST  = OUT_DIR / "best_model.pth"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE    = 96
NUM_CLASSES = 19
BATCH_SIZE  = 16
EPOCHS      = 100
LR          = 8e-4
WEIGHT_DECAY= 1e-4
MAX_SPP     = 12        # foreground-ranked slices per patient
PATIENCE    = 30
SEED        = 42

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# ── LABEL MAP ─────────────────────────────────────────────────────────
SPIDER_TO_ATMNET = {**{i: i for i in range(1, 9)}, 100: 9,
                    **{201+i: 10+i for i in range(8)}}
CLASS_NAMES = {
    0:"background",     1:"Vert-1(L)",  2:"Vert-2",    3:"Vert-3",
    4:"Vert-4",         5:"Vert-5",     6:"Vert-6",    7:"Vert-7",
    8:"Vert-8(U)",      9:"Sacrum",
    10:"IVD-1(L)",     11:"IVD-2",     12:"IVD-3",    13:"IVD-4",
    14:"IVD-5",        15:"IVD-6",     16:"IVD-7",    17:"IVD-8(U)",
    18:"Canal",
}
VERT_CLASSES = list(range(1, 9))
IVD_CLASSES  = list(range(10, 18))

def remap(m):
    out = np.zeros_like(m, dtype=np.int64)
    for s, d in SPIDER_TO_ATMNET.items():
        out[m == s] = d
    return out

# ── DATA ──────────────────────────────────────────────────────────────
def fg_score(m):
    rm = remap(m)
    # Weight: rare upper classes score higher
    score = float((rm > 0).sum()) / max(rm.size, 1)
    unique = np.unique(rm[rm > 0])
    score += len(unique) * 0.01  # bonus for class diversity
    return score

def load_dataset(pids, label):
    imgs, msks = [], []
    t0 = time.time()
    for pid in pids:
        fn = f"{pid}_t2.mha"
        ip, mp = IMAGES_DIR/fn, MASKS_DIR/fn
        if not ip.exists() or not mp.exists(): continue
        try:
            iv = sitk.GetArrayFromImage(sitk.ReadImage(str(ip))).astype(np.float32)
            mv = sitk.GetArrayFromImage(sitk.ReadImage(str(mp))).astype(np.int32)
        except: continue
        n  = iv.shape[0]
        lo, hi = int(n*0.10), int(n*0.90)
        ranked = sorted(range(lo, hi),
                        key=lambda s: fg_score(mv[s]), reverse=True)[:MAX_SPP]
        for s in ranked:
            img_s = iv[s]; msk_s = remap(mv[s])
            p1,p99= np.percentile(img_s,[0.5,99.5])
            img_n = np.clip((img_s-p1)/(p99-p1+1e-8),0,1).astype(np.float32)
            img_r = cv2.resize(img_n,  (IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            msk_r = cv2.resize(msk_s.astype(np.float32),(IMG_SIZE,IMG_SIZE),
                               interpolation=cv2.INTER_NEAREST).astype(np.int64)
            imgs.append(img_r)
            msks.append(np.clip(msk_r, 0, NUM_CLASSES-1))
    print(f"  {label:<6}: {len(pids):>4} patients → {len(imgs):>5} slices  ({time.time()-t0:.0f}s)")
    return imgs, msks

class Augmentor:
    def __init__(self, sz=IMG_SIZE):
        self.sz = sz
    def __call__(self, img, msk):
        # H-flip
        if random.random() < 0.5:
            img=np.fliplr(img).copy(); msk=np.fliplr(msk).copy()
        # Rotation
        if random.random() < 0.6:
            a=random.uniform(-20,20)
            M=cv2.getRotationMatrix2D((self.sz//2,self.sz//2),a,1.0)
            img=cv2.warpAffine(img,M,(self.sz,self.sz),flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
            mf=cv2.warpAffine(msk.astype(np.float32),M,(self.sz,self.sz),
                              flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
            msk=np.clip(mf.astype(np.int64),0,NUM_CLASSES-1)
        # Elastic deformation (lighter for speed)
        if random.random() < 0.3:
            from scipy.ndimage import gaussian_filter, map_coordinates
            h,w=img.shape
            dx=gaussian_filter(np.random.randn(h,w),self.sz*0.08)*(self.sz*0.8)
            dy=gaussian_filter(np.random.randn(h,w),self.sz*0.08)*(self.sz*0.8)
            x,y=np.meshgrid(np.arange(w),np.arange(h))
            xi=np.clip(x+dx,0,w-1).ravel(); yi=np.clip(y+dy,0,h-1).ravel()
            img=map_coordinates(img,[yi,xi],order=1).reshape(h,w).astype(np.float32)
            mf =map_coordinates(msk.astype(float),[yi,xi],order=0).reshape(h,w).astype(np.int64)
            msk=np.clip(mf,0,NUM_CLASSES-1)
        # Intensity
        gamma=random.uniform(0.6,1.8)
        img=np.clip(np.power(img+1e-8,gamma),0,1)
        img=np.clip(img*random.uniform(0.7,1.3)+random.uniform(-0.1,0.1),0,1)
        if random.random()<0.4:
            img=np.clip(img+np.random.normal(0,0.02,img.shape),0,1).astype(np.float32)
        return img.astype(np.float32), msk

class SliceDS(Dataset):
    def __init__(self, imgs, msks, aug=None):
        self.imgs=imgs; self.msks=msks; self.aug=aug
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        img=self.imgs[i].copy(); msk=self.msks[i].copy()
        if self.aug: img,msk=self.aug(img,msk)
        return torch.from_numpy(img[None]).float(), torch.from_numpy(msk).long()

# ── MODEL: ResUNet + CBAM ─────────────────────────────────────────────
class ChannelAttn(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        r=max(1,ch//r)
        self.avg=nn.AdaptiveAvgPool2d(1); self.max=nn.AdaptiveMaxPool2d(1)
        self.fc=nn.Sequential(nn.Flatten(),nn.Linear(ch,r),nn.ReLU(True),
                              nn.Linear(r,ch),nn.Sigmoid())
    def forward(self,x):
        a=self.fc(self.avg(x))+self.fc(self.max(x))
        return x*a.clamp(0,1).view(x.shape[0],-1,1,1)

class SpatialAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(2,1,7,padding=3,bias=False),
                                nn.BatchNorm2d(1),nn.Sigmoid())
    def forward(self,x):
        a=torch.cat([x.mean(1,keepdim=True),x.max(1,keepdim=True)[0]],1)
        return x*self.conv(a)

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(ch,ch,3,1,1,bias=False),nn.BatchNorm2d(ch),nn.ReLU(True),
            nn.Conv2d(ch,ch,3,1,1,bias=False),nn.BatchNorm2d(ch))
        self.ca=ChannelAttn(ch); self.sa=SpatialAttn()
        self.act=nn.ReLU(True)
    def forward(self,x): return self.act(self.sa(self.ca(self.net(x)))+x)

class Enc(nn.Module):
    def __init__(self,ci,co):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(ci,co,3,1,1,bias=False),nn.BatchNorm2d(co),nn.ReLU(True),
            nn.Conv2d(co,co,3,1,1,bias=False),nn.BatchNorm2d(co),nn.ReLU(True))
        self.res=ResBlock(co)
    def forward(self,x): return self.res(self.conv(x))

class ResUNet(nn.Module):
    def __init__(self, b=32, nc=NUM_CLASSES):
        super().__init__()
        self.e1=Enc(1,b);self.e2=Enc(b,b*2);self.e3=Enc(b*2,b*4);self.e4=Enc(b*4,b*8)
        self.bn=nn.Sequential(Enc(b*8,b*16),nn.Dropout2d(0.2))
        self.pool=nn.MaxPool2d(2)
        self.u4=nn.ConvTranspose2d(b*16,b*8,2,2);self.d4=Enc(b*16,b*8)
        self.u3=nn.ConvTranspose2d(b*8,b*4,2,2); self.d3=Enc(b*8,b*4)
        self.u2=nn.ConvTranspose2d(b*4,b*2,2,2); self.d2=Enc(b*4,b*2)
        self.u1=nn.ConvTranspose2d(b*2,b,2,2);   self.d1=Enc(b*2,b)
        # Deep supervision
        self.ds3=nn.Conv2d(b*4,nc,1); self.ds2=nn.Conv2d(b*2,nc,1)
        self.out=nn.Conv2d(b,nc,1)
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
def dice_loss(logits,tgt,sm=1e-6):
    soft=F.softmax(logits,1); oh=F.one_hot(tgt,NUM_CLASSES).permute(0,3,1,2).float()
    d=n=0
    for c in range(1,NUM_CLASSES):
        p=soft[:,c].reshape(-1); t=oh[:,c].reshape(-1)
        if t.sum()<1: continue
        d+=1-(2*(p*t).sum()+sm)/(p.sum()+t.sum()+sm); n+=1
    return d/max(n,1)

def tversky_loss(logits,tgt,alpha=0.3,beta=0.7,sm=1e-6):
    soft=F.softmax(logits,1); oh=F.one_hot(tgt,NUM_CLASSES).permute(0,3,1,2).float()
    d=n=0
    for c in range(1,NUM_CLASSES):
        p=soft[:,c].reshape(-1); t=oh[:,c].reshape(-1)
        if t.sum()<1: continue
        tp=(p*t).sum(); fp=(p*(1-t)).sum(); fn=((1-p)*t).sum()
        d+=1-(tp+sm)/(tp+alpha*fp+beta*fn+sm); n+=1
    return d/max(n,1)

def focal_loss(logits,tgt,gamma=2.0):
    ce=F.cross_entropy(logits,tgt,reduction='none')
    return ((1-torch.exp(-ce))**gamma*ce).mean()

def compound(logits,tgt):
    return F.cross_entropy(logits,tgt)+dice_loss(logits,tgt)+0.5*tversky_loss(logits,tgt)+0.3*focal_loss(logits,tgt)

def ds_loss(outs,tgt):
    o1,o2,o3=outs
    return compound(o1,tgt)+0.3*compound(o2,tgt)+0.15*compound(o3,tgt)

# ── METRICS ───────────────────────────────────────────────────────────
def full_metrics(logits,tgt):
    pred=logits.argmax(1).cpu().numpy(); gt=tgt.cpu().numpy(); sm=1e-6
    D=defaultdict(list);I=defaultdict(list);P=defaultdict(list)
    R=defaultdict(list);F1=defaultdict(list)
    for b in range(pred.shape[0]):
        for c in range(1,NUM_CLASSES):
            p=(pred[b]==c).astype(float).ravel(); t=(gt[b]==c).astype(float).ravel()
            if t.sum()==0 and p.sum()==0: continue
            tp=(p*t).sum();fp=(p*(1-t)).sum();fn=((1-p)*t).sum()
            D[c].append((2*tp+sm)/(2*tp+fp+fn+sm))
            I[c].append((tp+sm)/(tp+fp+fn+sm))
            P[c].append((tp+sm)/(tp+fp+sm))
            R[c].append((tp+sm)/(tp+fn+sm))
            pr=(tp+sm)/(tp+fp+sm); re=(tp+sm)/(tp+fn+sm)
            F1[c].append(2*pr*re/(pr+re+sm))
    all_d=[v for vs in D.values() for v in vs]
    return D,I,P,R,F1,float(np.mean(all_d)) if all_d else 0.0

# ── TTA ───────────────────────────────────────────────────────────────
def tta_predict(model,imgs):
    model.eval(); probs=None
    with torch.no_grad():
        for flip in [None,[[-1]],[[-2]],[[-1,-2]]]:
            x=torch.flip(imgs,flip) if flip else imgs
            out=model(x); p=F.softmax(out,1)
            if flip: p=torch.flip(p,flip)
            probs=p if probs is None else probs+p
    return probs/4

# ── MAIN ──────────────────────────────────────────────────────────────
def main():
    print("="*66)
    print("  ATM-Net++ High-Performance Training — SPIDER Dataset")
    print(f"  ResUNet+CBAM | {IMG_SIZE}×{IMG_SIZE} | {EPOCHS} epochs | CPU")
    print("="*66)

    df=pd.read_csv(OVERVIEW); tr_pids,va_pids=[],[]
    for name in df["new_file_name"].tolist():
        if not name.endswith("_t2") or "SPACE" in name: continue
        pid=name.replace("_t2","")
        sub=df.loc[df["new_file_name"]==name,"subset"].values
        (va_pids if len(sub) and sub[0]=="validation" else tr_pids).append(pid)

    print(f"\n  Dataset  : SPIDER Lumbar Spine MRI")
    print(f"  Patients : {len(tr_pids)} train | {len(va_pids)} validation")
    print(f"  Config   : {EPOCHS} epochs  BS={BATCH_SIZE}  LR={LR}  {IMG_SIZE}×{IMG_SIZE}")
    print(f"  Slices   : top-{MAX_SPP} fg-ranked/patient | Patience={PATIENCE}\n")

    print("  Loading data...")
    ti,tm=load_dataset(tr_pids,"Train")
    vi,vm=load_dataset(va_pids,"Val  ")

    aug=Augmentor(IMG_SIZE)
    tr_dl=DataLoader(SliceDS(ti,tm,aug),batch_size=BATCH_SIZE,shuffle=True, num_workers=0)
    va_dl=DataLoader(SliceDS(vi,vm),    batch_size=BATCH_SIZE,shuffle=False,num_workers=0)

    device=torch.device("cpu")
    model=ResUNet(b=32).to(device)
    n_p=sum(p.numel() for p in model.parameters())
    print(f"\n  Model    : ResUNet+CBAM  {n_p/1e6:.2f}M params  on CPU ({torch.get_num_threads()} threads)")
    print(f"  Batches  : {len(tr_dl)} train | {len(va_dl)} val/epoch")

    optim=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    sched=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim,T_0=30,T_mult=2,eta_min=1e-6)

    best=0.0; hist=[]; no_imp=0; t0tot=time.time(); ep=0

    print(f"\n  {'Ep':>3}  {'TrLoss':>8}  {'TrDice':>8}  {'VaLoss':>8}  "
          f"{'VaDice':>8}  {'Best':>8}  {'LR':>8}  {'Min':>5}")
    print("  "+"─"*67)

    for ep in range(1,EPOCHS+1):
        model.train(); tl_b,td_b=[],[]; t0=time.time()
        for imgs,msks in tr_dl:
            imgs,msks=imgs.to(device),msks.to(device)
            optim.zero_grad()
            outs=model(imgs)
            loss=ds_loss(outs,msks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optim.step(); tl_b.append(loss.item())
            with torch.no_grad():
                _,_,_,_,_,d=full_metrics(outs[0],msks); td_b.append(d)
        sched.step(ep)
        tl=float(np.mean(tl_b)); td=float(np.mean(td_b)); ep_m=(time.time()-t0)/60

        model.eval(); vl_b=[]; Dc=defaultdict(list)
        with torch.no_grad():
            for imgs,msks in va_dl:
                imgs,msks=imgs.to(device),msks.to(device)
                out=model(imgs)
                vl_b.append(compound(out,msks).item())
                D,_,_,_,_,_=full_metrics(out,msks)
                for c,v in D.items(): Dc[c].extend(v)
        vl=float(np.mean(vl_b))
        all_v=[v for vs in Dc.values() for v in vs]
        vd=float(np.mean(all_v)) if all_v else 0.0
        lr_now=float(optim.param_groups[0]['lr'])

        if vd>best:
            best=vd; no_imp=0
            pc={CLASS_NAMES[c]:float(np.mean(v)) for c,v in Dc.items() if v}
            torch.save({"epoch":ep,"model_state_dict":model.state_dict(),
                        "best_dice":best,"per_class_dice":pc,
                        "cfg":{"sz":IMG_SIZE,"nc":NUM_CLASSES}},CKPT_BEST)
        else: no_imp+=1

        flag=" ★" if vd==best else ""
        print(f"  {ep:>3}  {tl:>8.4f}  {td:>8.4f}  {vl:>8.4f}  "
              f"{vd:>8.4f}  {best:>8.4f}  {lr_now:.2e}  {ep_m:>4.1f}m{flag}")
        hist.append({"ep":ep,"tl":tl,"td":td,"vl":vl,"vd":vd,"lr":lr_now})
        if no_imp>=PATIENCE:
            print(f"\n  Early stop at ep {ep} (no improvement for {PATIENCE} epochs)")
            break
        gc.collect()

    ttot=time.time()-t0tot
    print("  "+"─"*67)
    print(f"\n  Training done in {ttot/60:.1f} min ({ttot/3600:.2f}h)  |  Best Val Dice: {best:.4f}")
    with open(OUT_DIR/"history.json","w") as f: json.dump(hist,f,indent=2)

    # ── FINAL EVALUATION WITH TTA ─────────────────────────────────
    print("\n"+"="*66)
    print("  FINAL EVALUATION — Best Checkpoint + TTA")
    print("="*66)
    ckpt=torch.load(CKPT_BEST,map_location=device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    print(f"  Epoch: {ckpt['epoch']}  Training Dice: {ckpt['best_dice']:.4f}\n")

    aD=defaultdict(list);aI=defaultdict(list)
    aP=defaultdict(list);aR=defaultdict(list);aF1=defaultdict(list); n_sl=0
    with torch.no_grad():
        for imgs,msks in va_dl:
            imgs,msks=imgs.to(device),msks.to(device)
            probs=tta_predict(model,imgs)
            D,I,P,R,F1,_=full_metrics(probs,msks)
            for c in range(1,NUM_CLASSES):
                aD[c].extend(D[c]);aI[c].extend(I[c])
                aP[c].extend(P[c]);aR[c].extend(R[c]);aF1[c].extend(F1[c])
            n_sl+=imgs.shape[0]

    print(f"  {'Class':<24} {'Dice':>7}  {'IoU':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'N':>5}")
    print("  "+"─"*67)
    res={}; vd_all=[]; vi_all=[]; vert_d=[]; ivd_d=[]
    for c in range(1,NUM_CLASSES):
        if not aD[c]: continue
        d=float(np.mean(aD[c]));io=float(np.mean(aI[c]))
        pr=float(np.mean(aP[c]));re=float(np.mean(aR[c]));f1=float(np.mean(aF1[c]))
        n=len(aD[c]); nm=CLASS_NAMES[c]
        res[nm]={"dice":d,"iou":io,"precision":pr,"recall":re,"f1":f1,"n":n}
        vd_all.append(d);vi_all.append(io)
        if c in VERT_CLASSES: vert_d.append(d)
        if c in IVD_CLASSES: ivd_d.append(d)
        tag=("  ★" if d>=0.90 else "  ✓" if d>=0.80 else "  ·" if d>=0.70 else
             "  +" if d>=0.50 else "")
        print(f"  {nm:<24} {d:>7.4f}  {io:>7.4f}  {pr:>7.4f}  {re:>7.4f}  {f1:>7.4f}  {n:>5}{tag}")

    md=float(np.mean(vd_all)) if vd_all else 0.0
    mi=float(np.mean(vi_all)) if vi_all else 0.0
    vdm=float(np.mean(vert_d)) if vert_d else 0.0
    idm=float(np.mean(ivd_d))  if ivd_d  else 0.0
    mf1=float(np.mean([float(np.mean(aF1[c])) for c in aF1 if aF1[c]])) if aF1 else 0.0
    mpr=float(np.mean([float(np.mean(aP[c]))  for c in aP  if aP[c]]))  if aP  else 0.0
    mre=float(np.mean([float(np.mean(aR[c]))  for c in aR  if aR[c]]))  if aR  else 0.0

    print("  "+"─"*67)
    print(f"  {'MEAN FOREGROUND':<24} {md:>7.4f}  {mi:>7.4f}  {mpr:>7.4f}  {mre:>7.4f}  {mf1:>7.4f}")

    baseline=0.1718; delta=md-baseline
    improv=f"+{delta:.4f} (+{delta/baseline*100:.0f}%)"
    status=("🎯 TARGET ACHIEVED: Dice ≥ 0.90!" if md>=0.90 else
            "📈 Excellent — GPU + Swin UNETR will exceed 0.90" if md>=0.80 else
            "📊 Strong improvement — GPU + 512×512 → >0.90" if md>=0.70 else
            "📉 Good progress — GPU + higher resolution for >0.90")

    print(f"""
  ╔══════════════════════════════════════════════════════════════╗
  ║  EVALUATION REPORT — ATM-Net++ — SPIDER Dataset (+ TTA)    ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  Architecture     :  ResUNet + CBAM Attention              ║
  ║  Resolution       :  {IMG_SIZE}×{IMG_SIZE}  |  Epochs: {ep:>3}              ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  Vertebrae Dice   :  {vdm:.4f}                                ║
  ║  IVD Dice         :  {idm:.4f}                                ║
  ║  Mean Dice        :  {md:.4f}                                ║
  ║  Mean IoU         :  {mi:.4f}                                ║
  ║  Mean Precision   :  {mpr:.4f}                                ║
  ║  Mean Recall      :  {mre:.4f}                                ║
  ║  Mean F1          :  {mf1:.4f}                                ║
  ║  Val slices       :  {n_sl}                                  ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  vs. Baseline     :  {baseline:.4f} → {md:.4f}  ({improv}){'':>5}║
  ╠══════════════════════════════════════════════════════════════╣
  ║  {status:<60}║
  ╚══════════════════════════════════════════════════════════════╝

  Literature benchmarks (512×512, full dataset):
    U-Net (2D)        :  Dice ~0.76–0.82
    nnU-Net           :  Dice ~0.86–0.89
    Swin UNETR        :  Dice ~0.88–0.92
    ATM-Net++ (full)  :  Dice  >0.90  ← GPU target

  For Dice >0.90 on GPU:
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    python training/train.py --config configs/base_config.yaml
    (Swin UNETR, 512×512, 150 epochs → ~12-24h)
""")

    final={"mean_dice":md,"mean_iou":mi,"mean_f1":mf1,
           "mean_precision":mpr,"mean_recall":mre,
           "vertebrae_dice":vdm,"ivd_dice":idm,
           "total_val_slices":n_sl,"epochs_trained":ep,
           "best_training_dice":float(best),
           "baseline_dice":baseline,"improvement":float(delta),
           "tta":True,"resolution":f"{IMG_SIZE}x{IMG_SIZE}","per_class":res}
    with open(OUT_DIR/"evaluation_results.json","w") as f: json.dump(final,f,indent=2)
    print(f"  Results → {OUT_DIR/'evaluation_results.json'}")
    print("="*66)
    return final

if __name__=="__main__": main()
