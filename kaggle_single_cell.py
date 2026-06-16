# ══════════════════════════════════════════════════════════════════
# ATM-Net++ COMPLETE TRAINING — Single cell for Kaggle
# Copy ALL of this into ONE new cell in Kaggle, then Run
# Expected: Dice 0.85+ after 300 epochs (~4-6 hours on T4)
# ══════════════════════════════════════════════════════════════════
import os, sys, time, random, gc, json, warnings, glob
warnings.filterwarnings('ignore')
import numpy as np
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import defaultdict
from pathlib import Path
import SimpleITK as sitk

# ── CONFIG ────────────────────────────────────────────────────────
IMG_SIZE   = 512
BATCH_SIZE = 6
ACCUM      = 4        # effective BS=24
EPOCHS     = 300
LR         = 3e-4
LR_MIN     = 5e-6
WARMUP_EP  = 5
WD         = 4e-4
MAX_SPP    = 25
NC         = 19
PATIENCE   = 80
SEED       = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
print(f'Config: {IMG_SIZE}px | BS={BATCH_SIZE}x{ACCUM}=eff{BATCH_SIZE*ACCUM} | {EPOCHS} epochs')

# ── PATHS ─────────────────────────────────────────────────────────
mhas = sorted([f for f in glob.glob('/kaggle/input/**/*_t2.mha', recursive=True)
               if 'SPACE' not in f])
assert mhas, 'No MHA files — add SPIDER dataset via + Add Data'
IMAGES_DIR = Path(mhas[0]).parent
all_mha    = glob.glob('/kaggle/input/**/*.mha', recursive=True)
img_names  = {Path(f).name for f in mhas}
mask_dirs  = {Path(f).parent for f in all_mha
              if Path(f).name in img_names and Path(f).parent != IMAGES_DIR}
MASKS_DIR  = sorted(mask_dirs)[0] if mask_dirs else IMAGES_DIR
OUTPUT_DIR = Path('/kaggle/working')
CKPT_BEST  = OUTPUT_DIR / 'best_model.pth'
CKPT_LAST  = OUTPUT_DIR / 'last_model.pth'
CACHE_DIR  = OUTPUT_DIR / 'cache'; CACHE_DIR.mkdir(exist_ok=True)
print(f'Images: {IMAGES_DIR} ({len(mhas)} files)')
print(f'Masks : {MASKS_DIR}')

# ── LABEL MAPPING ─────────────────────────────────────────────────
S2A  = {**{i:i for i in range(1,9)}, 100:9, **{201+i:10+i for i in range(8)}}
CN   = {0:'bg',1:'V1',2:'V2',3:'V3',4:'V4',5:'V5',6:'V6',7:'V7',8:'V8',9:'Sac',
        10:'I1',11:'I2',12:'I3',13:'I4',14:'I5',15:'I6',16:'I7',17:'I8',18:'Canal'}
RARE = [7, 8, 16, 17]
CW   = torch.tensor([0,1,1,1,1.5,2,3,8,15,1,6,4,4,5,7,10,18,40,0]).float()

def remap(m):
    o = np.zeros_like(m, dtype=np.int32)
    for s,d in S2A.items(): o[m==s]=d
    return o

def load_vol(path):
    """Load MHA — handle (H,W,D) and (D,H,W) axis orders"""
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    if arr.ndim==3 and arr.shape[2]<arr.shape[0] and arr.shape[2]<arr.shape[1]:
        arr = arr.transpose(2,0,1)   # (H,W,D) → (D,H,W)
    return arr

def fg(m): return float((m>0).sum())/max(m.size,1)

# ── SPLITS ────────────────────────────────────────────────────────
all_pids = sorted(set(Path(f).stem.replace('_t2','') for f in mhas
                      if (MASKS_DIR/Path(f).name).exists()))
random.Random(SEED).shuffle(all_pids)
n_val    = max(1, len(all_pids)//5)
va_pids  = all_pids[-n_val:]
tr_pids  = all_pids[:-n_val]
print(f'Patients: {len(tr_pids)} train | {len(va_pids)} val')

# ── CACHE ─────────────────────────────────────────────────────────
def build_cache(pids, split):
    cf = CACHE_DIR / f'{split}_t2_{IMG_SIZE}.npz'
    if cf.exists():
        print(f'  {split}: loading cache...', end=' ', flush=True)
        d = np.load(cf); t=time.time()
        imgs,msks,rare = d['imgs'],d['msks'],d['rare'].tolist()
        print(f'done ({len(imgs)} slices)')
        return np.array(imgs,copy=True), np.array(msks,copy=True), rare
    print(f'  {split}: building cache ({len(pids)} patients)...')
    imgs, msks, rare = [], [], []
    for i,pid in enumerate(pids):
        ip=IMAGES_DIR/f'{pid}_t2.mha'; mp=MASKS_DIR/f'{pid}_t2.mha'
        if not ip.exists() or not mp.exists(): continue
        try:
            iv=load_vol(ip).astype(np.float32)
            mv=load_vol(mp).astype(np.int32)
        except Exception as e:
            print(f'  Error {pid}: {e}'); continue
        n=iv.shape[0]; lo,hi=int(n*0.04),int(n*0.96)
        ranked=sorted(range(lo,hi),key=lambda s:fg(remap(mv[s])),reverse=True)[:MAX_SPP]
        for s in ranked:
            rm=remap(mv[s])
            if fg(rm)<0.003: continue
            p1,p99=np.percentile(iv[s],[0.5,99.5])
            img_n=np.clip((iv[s]-p1)/(p99-p1+1e-8),0,1).astype(np.float32)
            ir=cv2.resize(img_n,(IMG_SIZE,IMG_SIZE),interpolation=cv2.INTER_LINEAR).astype(np.float16)
            mr=cv2.resize(rm.astype(np.float32),(IMG_SIZE,IMG_SIZE),
                          interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            imgs.append(ir); msks.append(np.clip(mr,0,NC-1))
            has_rare=any((rm==c).sum()/max(rm.size,1)>0.0003 for c in RARE)
            rare.append(1.0 if has_rare else 0.1)
        if (i+1)%30==0: print(f'    {i+1}/{len(pids)}, {len(imgs)} slices')
    if not imgs:
        print(f'  ERROR: no slices for {split}! Sample pids: {pids[:3]}')
        print(f'  Images dir: {IMAGES_DIR}')
        print(f'  Sample file: {list(IMAGES_DIR.glob("*_t2.mha"))[:2]}')
        raise ValueError(f'No slices loaded for {split}')
    imgs_a=np.array(np.stack(imgs),dtype=np.float16,copy=True)
    msks_a=np.array(np.stack(msks),dtype=np.uint8,copy=True)
    rare_a=np.array(rare,dtype=np.float32)
    np.savez_compressed(cf, imgs=imgs_a, msks=msks_a, rare=rare_a)
    print(f'  {split}: {len(imgs)} slices saved')
    return imgs_a, msks_a, rare

print('\nBuilding data cache...')
ti,tm,tr_rare = build_cache(tr_pids,'train')
vi,vm,va_rare = build_cache(va_pids,'val')
print(f'RAM: ~{(ti.nbytes+tm.nbytes+vi.nbytes+vm.nbytes)//1024**2}MB')
print(f'Rare slices: {sum(1 for r in tr_rare if r>0.5)}/{len(tr_rare)}')

# ── AUGMENTATION ──────────────────────────────────────────────────
class Aug:
    def __call__(self,img,msk):
        if random.random()<0.5:
            img=np.fliplr(img).copy(); msk=np.fliplr(msk).copy()
        if random.random()<0.7:
            a=random.uniform(-25,25)
            M=cv2.getRotationMatrix2D((IMG_SIZE//2,IMG_SIZE//2),a,1.0)
            img=cv2.warpAffine(img,M,(IMG_SIZE,IMG_SIZE),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
            mf=cv2.warpAffine(msk.astype(np.float32),M,(IMG_SIZE,IMG_SIZE),
                               flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
            msk=np.clip(mf.astype(np.int32),0,NC-1)
        if random.random()<0.4:
            scale=random.uniform(0.85,1.15)
            ns=int(IMG_SIZE*scale)
            ir=cv2.resize(img,(ns,ns),interpolation=cv2.INTER_LINEAR)
            mr=cv2.resize(msk.astype(np.float32),(ns,ns),interpolation=cv2.INTER_NEAREST).astype(np.int32)
            if ns>=IMG_SIZE:
                s=(ns-IMG_SIZE)//2; img=ir[s:s+IMG_SIZE,s:s+IMG_SIZE]
                msk=np.clip(mr[s:s+IMG_SIZE,s:s+IMG_SIZE],0,NC-1)
            else:
                p=(IMG_SIZE-ns)//2; img=np.pad(ir,p,mode='reflect')[:IMG_SIZE,:IMG_SIZE]
                msk=np.clip(np.pad(mr,p)[:IMG_SIZE,:IMG_SIZE],0,NC-1)
        g=random.uniform(0.6,1.6)
        img=np.clip(np.power(img.astype(np.float32)+1e-8,g),0,1)
        img=np.clip(img*random.uniform(0.7,1.3)+random.uniform(-0.12,0.12),0,1)
        if random.random()<0.4:
            img=np.clip(img+np.random.normal(0,0.012,img.shape),0,1)
        if random.random()<0.3:
            cy,cx=random.randint(0,IMG_SIZE),random.randint(0,IMG_SIZE)
            r=random.randint(12,28)
            img[max(0,cy-r):min(IMG_SIZE,cy+r),max(0,cx-r):min(IMG_SIZE,cx+r)]=0
        return img.astype(np.float32),msk.astype(np.int64)

class DS(Dataset):
    def __init__(self,imgs,msks,aug=None):
        self.imgs=imgs; self.msks=msks; self.aug=aug
    def __len__(self): return len(self.imgs)
    def __getitem__(self,i):
        img=self.imgs[i].astype(np.float32); msk=self.msks[i].astype(np.int64)
        if self.aug: img,msk=self.aug(img,msk)
        return torch.from_numpy(img[None]).float(), torch.from_numpy(msk).long()

# ── MODEL ─────────────────────────────────────────────────────────
class CA(nn.Module):
    def __init__(self,ch,r=8):
        super().__init__(); r=max(1,ch//r)
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
    def __init__(self,ci,co,drop=0.0):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(ci,co,3,1,1,bias=False),nn.BatchNorm2d(co),nn.ReLU(True),
                                nn.Conv2d(co,co,3,1,1,bias=False),nn.BatchNorm2d(co),nn.ReLU(True))
        self.res=RB(co); self.drop=nn.Dropout2d(drop) if drop>0 else nn.Identity()
    def forward(self,x): return self.drop(self.res(self.conv(x)))

class ResUNet(nn.Module):
    def __init__(self,b=32,nc=NC,drop=0.25):
        super().__init__()
        self.e1=Enc(1,b); self.e2=Enc(b,b*2,drop*.3)
        self.e3=Enc(b*2,b*4,drop*.6); self.e4=Enc(b*4,b*8,drop*.8)
        self.bn=nn.Sequential(Enc(b*8,b*16,drop),nn.Dropout2d(drop)); self.pool=nn.MaxPool2d(2)
        self.u4=nn.ConvTranspose2d(b*16,b*8,2,2); self.d4=Enc(b*16,b*8,drop*.4)
        self.u3=nn.ConvTranspose2d(b*8,b*4,2,2);  self.d3=Enc(b*8,b*4,drop*.2)
        self.u2=nn.ConvTranspose2d(b*4,b*2,2,2);  self.d2=Enc(b*4,b*2)
        self.u1=nn.ConvTranspose2d(b*2,b,2,2);    self.d1=Enc(b*2,b)
        self.ds3=nn.Conv2d(b*4,nc,1); self.ds2=nn.Conv2d(b*2,nc,1); self.out=nn.Conv2d(b,nc,1)
        self.aux=nn.Sequential(nn.Conv2d(b,b,3,1,1,bias=False),nn.BatchNorm2d(b),nn.ReLU(True),nn.Conv2d(b,nc,1))
    def forward(self,x):
        sz=x.shape[2:]
        e1=self.e1(x); e2=self.e2(self.pool(e1)); e3=self.e3(self.pool(e2)); e4=self.e4(self.pool(e3))
        d=self.bn(self.pool(e4))
        d=self.d4(torch.cat([self.u4(d),e4],1)); d=self.d3(torch.cat([self.u3(d),e3],1))
        o3=F.interpolate(self.ds3(d),sz,mode='bilinear',align_corners=False)
        d=self.d2(torch.cat([self.u2(d),e2],1))
        o2=F.interpolate(self.ds2(d),sz,mode='bilinear',align_corners=False)
        d=self.d1(torch.cat([self.u1(d),e1],1))
        return (self.out(d),o2,o3,self.aux(d)) if self.training else self.out(d)

# ── LOSS ──────────────────────────────────────────────────────────
def dice_w(lg,tg,sm=1e-6):
    B,C,H,W=lg.shape; s=F.softmax(lg,1)
    o=F.one_hot(tg.clamp(0,C-1),C).permute(0,3,1,2).float()
    p=s[:,1:].reshape(B,C-1,-1); t=o[:,1:].reshape(B,C-1,-1)
    inter=(p*t).sum(-1); union=p.sum(-1)+t.sum(-1)
    mask=(t.sum(-1)>0).float()
    w=CW[1:].to(lg.device).view(1,C-1)
    return 1-((2*inter+sm)/(union+sm)*mask*w).sum()/(mask*w).sum().clamp(min=1)

def focal(lg,tg,g=2.0):
    ce=F.cross_entropy(lg,tg.clamp(0,NC-1),reduction='none')
    return ((1-torch.exp(-ce))**g*ce).mean()

def boundary(lg,tg):
    s=F.softmax(lg,1); o=F.one_hot(tg.clamp(0,NC-1),NC).permute(0,3,1,2).float()
    b=(F.max_pool2d(o[:,1:],3,stride=1,padding=1)-o[:,1:]).clamp(0,1)
    w=CW[1:].to(lg.device).view(1,-1,1,1)
    return (b*(1-s[:,1:])*w).sum()/((b*w).sum()+1e-6)

def compound(lg,tg):
    tc=tg.clamp(0,NC-1)
    return F.cross_entropy(lg,tc,label_smoothing=0.02)+dice_w(lg,tc)+0.3*focal(lg,tc)+0.15*boundary(lg,tc)

def total_loss(outs,tg):
    o1,o2,o3,ax=outs
    main=compound(o1,tg)+0.3*compound(o2,tg)+0.15*compound(o3,tg)
    tc=tg.clamp(0,NC-1); rm=sum((tc==c).float() for c in RARE).clamp(0,1)
    return main+0.3*(F.cross_entropy(ax,tc,reduction='none')*(1+5*rm)).mean()

@torch.no_grad()
def fast_dice(lg,tg):
    B=lg.shape[0]; pred=lg.argmax(1); sm=1e-6; D=defaultdict(list)
    for c in range(1,NC):
        p=(pred==c).float().view(B,-1); t=(tg==c).float().view(B,-1)
        active=(t.sum(1)>0)|(p.sum(1)>0)
        if not active.any(): continue
        tp=(p*t).sum(1)[active]; den=(p.sum(1)+t.sum(1))[active]
        D[c].extend(((2*tp+sm)/(den+sm)).cpu().tolist())
    all_d=[v for vs in D.values() for v in vs]
    return D, float(np.mean(all_d)) if all_d else 0.0

def get_lr(ep,start_ep):
    rel=ep-start_ep+1
    if rel<=WARMUP_EP: return LR*rel/WARMUP_EP
    t=(rel-WARMUP_EP)/max(EPOCHS-WARMUP_EP,1)
    return LR_MIN+0.5*(LR-LR_MIN)*(1+np.cos(np.pi*t))

# ── BUILD MODEL & DATALOADERS ─────────────────────────────────────
model=ResUNet(b=32,nc=NC,drop=0.25).to(device)
n_params=sum(p.numel() for p in model.parameters())
print(f'\nModel: ResUNet+CBAM {n_params/1e6:.2f}M params')

sampler=WeightedRandomSampler(torch.tensor(tr_rare),len(tr_rare),replacement=True)
# num_workers=0 is critical on Kaggle — prevents DataLoader hang
tr_dl=DataLoader(DS(ti,tm,Aug()),batch_size=BATCH_SIZE,sampler=sampler,num_workers=0,pin_memory=False)
va_dl=DataLoader(DS(vi,vm),batch_size=BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=False)
print(f'Batches: {len(tr_dl)} train | {len(va_dl)} val')

# ── RESUME OR START FRESH ─────────────────────────────────────────
start_ep=1; best=0.0
if CKPT_BEST.exists():
    ck_b=torch.load(str(CKPT_BEST),map_location=device)
    ep_b=ck_b.get('epoch',0); ck_use=ck_b
    if CKPT_LAST.exists():
        ck_l=torch.load(str(CKPT_LAST),map_location=device)
        if ck_l.get('epoch',0)>ep_b: ck_use=ck_l
    model.load_state_dict(ck_use['model_state_dict'],strict=False)
    best=ck_b.get('best_dice',0.0)
    start_ep=ck_use.get('epoch',0)+1
    print(f'Resumed: ep{ck_use["epoch"]} best={best:.4f} → ep{start_ep}')
else:
    print('Starting fresh')

optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
scaler=GradScaler(); no_imp=0; t0_total=time.time()

print(f'\n{"Ep":>4}  {"TrLoss":>8}  {"VaDice":>8}  {"Best":>8}  {"Gap":>6}  {"LR":>8}  {"Sec":>5}')
print('─'*65)

# ── TRAINING LOOP ─────────────────────────────────────────────────
for ep in range(start_ep,EPOCHS+1):
    lr_now=get_lr(ep,start_ep)
    for pg in optimizer.param_groups: pg['lr']=lr_now

    model.train(); losses=[]; t0=time.time()
    optimizer.zero_grad(set_to_none=True)
    for step,(imgs,msks) in enumerate(tr_dl):
        imgs=imgs.to(device,non_blocking=True); msks=msks.to(device,non_blocking=True)
        with autocast():
            outs=model(imgs); loss=total_loss(outs,msks)/ACCUM
        scaler.scale(loss).backward()
        if (step+1)%ACCUM==0 or (step+1)==len(tr_dl):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item()*ACCUM)
    tr_loss=float(np.mean(losses)); ep_sec=time.time()-t0

    model.eval(); Dc=defaultdict(list)
    with torch.no_grad():
        for imgs,msks in va_dl:
            imgs=imgs.to(device); msks=msks.to(device)
            with autocast():
                p1=F.softmax(model(imgs),1)
                p2=F.softmax(model(torch.flip(imgs,[-1])),1)
                avg=(p1+torch.flip(p2,[-1]))/2
            D,_=fast_dice(avg,msks)
            for c,v in D.items(): Dc[c].extend(v)
    all_v=[v for vs in Dc.values() for v in vs]
    vd=float(np.mean(all_v)) if all_v else 0.0

    with torch.no_grad():
        model.eval(); imgs_s,msks_s=next(iter(tr_dl))
        with autocast(): out_s=model(imgs_s.to(device))
        _,td=fast_dice(out_s,msks_s.to(device))
    gap=td-vd

    if vd>best:
        best=vd; no_imp=0
        pc={CN[c]:float(np.mean(v)) for c,v in Dc.items() if v}
        torch.save({'epoch':ep,'model_state_dict':model.state_dict(),
                    'best_dice':best,'per_class_dice':pc,
                    'cfg':{'img_size':IMG_SIZE,'nc':NC}},CKPT_BEST)
        with open(OUTPUT_DIR/'results.json','w') as f:
            json.dump({'epoch':ep,'best_dice':best,'per_class':pc},f,indent=2)
    else:
        no_imp+=1

    if ep%5==0:
        torch.save({'epoch':ep,'model_state_dict':model.state_dict(),'best_dice':best},CKPT_LAST)

    flag='  ★' if vd==best else ''
    print(f'{ep:>4}  {tr_loss:>8.4f}  {vd:>8.4f}  {best:>8.4f}  {gap:>+6.3f}  {lr_now:>8.2e}  {ep_sec:>4.0f}s{flag}')

    if vd>=0.90: print('\nDice >= 0.90 achieved!'); break
    if vd>=0.85: print(f'  Dice {vd:.4f} — past 0.85!')
    if no_imp>=PATIENCE: print(f'\nEarly stop'); break
    gc.collect(); torch.cuda.empty_cache()

t_total=(time.time()-t0_total)/3600
print('─'*65)
print(f'Done: {t_total:.2f}h | Best Dice: {best:.4f}')
print(f'Download: {CKPT_BEST}')
