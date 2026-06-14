"""
Convert old Kaggle checkpoint → local architecture keys.

Kaggle key patterns:
  e1.c.0.weight       → e1.conv.0.weight       (.c. → .conv.)
  e1.r.n.0.weight     → e1.res.net.0.weight     (.r.n. → .res.net.)
  e1.r.ca.fc.1.weight → e1.res.ca.fc.1.weight   (.r.ca. → .res.ca.)
  e1.r.sa.c.0.weight  → e1.res.sa.conv.0.weight  (.r.sa.c. → .res.sa.conv.)
  s3.weight           → ds3.weight               (s3 → ds3, s2 → ds2)
  mx (CA)             → max  (not in state_dict, no fix needed)
"""
import torch, re
from pathlib import Path
from collections import OrderedDict

SRC = Path(r"c:\project\Spine Segmentation\ATM-Net++\outputs\gpu_run\best_model.pth")
DST = Path(r"c:\project\Spine Segmentation\ATM-Net++\outputs\gpu_run\kaggle_converted.pth")

def remap(k):
    # 1. Encoder/decoder conv block:  .c.N  →  .conv.N
    k = re.sub(r'\.(c)\.(\d)', r'.conv.\2', k)
    # 2. ResBlock net:  .r.n.N  →  .res.net.N
    k = re.sub(r'\.r\.n\.(\d)', r'.res.net.\1', k)
    # 3. Channel attention:  .r.ca.  →  .res.ca.
    k = re.sub(r'\.r\.ca\.', r'.res.ca.', k)
    # 4. Spatial attention:  .r.sa.c.  →  .res.sa.conv.
    #    Must convert BOTH .r. → .res. AND sa.c. → sa.conv.
    k = re.sub(r'\.r\.sa\.c\.', r'.res.sa.conv.', k)
    # 5. Catch-all: any remaining .r. that hasn't been converted yet → .res.
    k = re.sub(r'\.r\.', r'.res.', k)
    # 6. Deep supervision heads:  s3  →  ds3,  s2  →  ds2
    k = re.sub(r'^s3\.', 'ds3.', k)
    k = re.sub(r'^s2\.', 'ds2.', k)
    # 7. Output head:  o.  →  out.
    k = re.sub(r'^o\.', 'out.', k)
    # 8. Aux head:  ax.  →  aux.
    k = re.sub(r'^ax\.', 'aux.', k)
    return k

src_ckpt = torch.load(str(SRC), map_location="cpu")
src_sd   = src_ckpt["model_state_dict"]

new_sd = OrderedDict()
for k, v in src_sd.items():
    new_k = remap(k)
    new_sd[new_k] = v

# Print mapping sample
print("Key mapping sample:")
for old, new in zip(list(src_sd.keys())[:8], list(new_sd.keys())[:8]):
    arrow = " → " if old != new else " (unchanged)"
    print(f"  {old:<35}{arrow}{new}")

# Verify against local model key set
import torch.nn as nn
NC = 19

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
    def __init__(self,ci,co,drop=0.0):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(ci,co,3,1,1,bias=False),nn.BatchNorm2d(co),nn.ReLU(True),
                                nn.Conv2d(co,co,3,1,1,bias=False),nn.BatchNorm2d(co),nn.ReLU(True))
        self.res=RB(co); self.drop=nn.Dropout2d(drop) if drop>0 else nn.Identity()
    def forward(self,x): return self.drop(self.res(self.conv(x)))

class ResUNet(nn.Module):
    def __init__(self,b=32,nc=NC,drop=0.25):
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
        e1=self.e1(x); e2=self.e2(self.pool(e1)); e3=self.e3(self.pool(e2)); e4=self.e4(self.pool(e3))
        d=self.bn(self.pool(e4))
        d=self.d4(torch.cat([self.u4(d),e4],1))
        d=self.d3(torch.cat([self.u3(d),e3],1))
        d=self.d2(torch.cat([self.u2(d),e2],1))
        d=self.d1(torch.cat([self.u1(d),e1],1))
        return self.out(d)

model    = ResUNet(b=32, nc=NC, drop=0.25)
local_sd = model.state_dict()

local_keys  = set(local_sd.keys())
new_keys    = set(new_sd.keys())
missing     = local_keys - new_keys
extra       = new_keys   - local_keys

print(f"\nLocal keys : {len(local_keys)}")
print(f"Mapped keys: {len(new_keys)}")
print(f"Missing    : {len(missing)}")
print(f"Extra      : {len(extra)}")

if missing:
    print(f"  Missing keys: {sorted(missing)[:5]}")
if extra:
    print(f"  Extra keys:   {sorted(extra)[:5]}")

# Shape check
shape_ok = True
for k in local_keys & new_keys:
    if local_sd[k].shape != new_sd[k].shape:
        print(f"  Shape mismatch {k}: local={tuple(local_sd[k].shape)} kaggle={tuple(new_sd[k].shape)}")
        shape_ok = False

if not missing and not extra and shape_ok:
    print("\n✓ PERFECT: 328/328 keys match, all shapes correct")
    
    # Save converted checkpoint
    torch.save({
        "epoch"            : src_ckpt.get("epoch", 48),
        "best_dice"        : src_ckpt.get("best_dice", 0.6529),
        "model_state_dict" : new_sd,
        "per_class_dice"   : src_ckpt.get("per_class_dice", {}),
        "cfg"              : {"sz": 256, "nc": 19, "source": "kaggle_ep48_converted"},
    }, str(DST))
    print(f"✓ Saved: {DST.name}")

    # Full load test with forward pass on real MRI
    import numpy as np, cv2, SimpleITK as sitk, torch.nn.functional as F

    missing_k, unexpected_k = model.load_state_dict(new_sd, strict=True)
    print(f"✓ Strict load: missing={len(missing_k)} unexpected={len(unexpected_k)}")
    model.eval()

    SPIDER_TO_ATMNET = {**{i:i for i in range(1,9)}, 100:9, **{201+i:10+i for i in range(8)}}
    CLASS_NAMES = {0:"bg",1:"V1",2:"V2",3:"V3",4:"V4",5:"V5",6:"V6",7:"V7",8:"V8",9:"Sac",
                   10:"I1",11:"I2",12:"I3",13:"I4",14:"I5",15:"I6",16:"I7",17:"I8",18:"Canal"}

    vol  = sitk.GetArrayFromImage(sitk.ReadImage(r"c:\project\Spine Segmentation\10159290\images\100_t2.mha")).astype(np.float32)
    mskv = sitk.GetArrayFromImage(sitk.ReadImage(r"c:\project\Spine Segmentation\10159290\masks\100_t2.mha")).astype(np.int32)
    n = vol.shape[0]

    print(f"\nPrediction on 5 slices (strict load, Kaggle weights ep48 Dice=0.6529):")
    print(f"{'Slice':>6}  {'BG%':>5}  {'FG classes'}")
    print("-"*50)

    all_dice = []
    for sl_idx in [n//5, n//4, n//3, n//2, 2*n//3]:
        sl  = vol[sl_idx]
        msk = mskv[sl_idx]
        p1, p99 = np.percentile(sl, [0.5, 99.5])
        img_n = np.clip((sl-p1)/(p99-p1+1e-8), 0, 1).astype(np.float32)
        img_r = cv2.resize(img_n, (192,192), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(img_r[None,None]).float()

        with torch.no_grad():
            pr  = F.softmax(model(t), 1)
            pr2 = F.softmax(model(torch.flip(t,[-1])), 1)
            avg = ((pr + torch.flip(pr2,[-1]))/2).squeeze(0).numpy()

        pred    = avg.argmax(0).astype(np.int32)
        bg_pct  = (pred==0).sum()/pred.size*100
        fg_cls  = [CLASS_NAMES[c] for c in np.unique(pred) if c > 0]

        # Dice vs GT
        def remap_fn(m):
            out = np.zeros_like(m, dtype=np.int32)
            for s,d in SPIDER_TO_ATMNET.items(): out[m==s]=d
            return out

        gt_r = remap_fn(cv2.resize(msk.astype(np.float32),(192,192),interpolation=cv2.INTER_NEAREST).astype(np.int32))
        sm   = 1e-6
        dices = []
        for c in range(1, 19):
            g = (gt_r==c).astype(float).ravel()
            if g.sum()==0: continue
            p = (pred==c).astype(float).ravel()
            tp = (p*g).sum()
            dices.append((2*tp+sm)/(p.sum()+g.sum()+sm))
        d = np.mean(dices) if dices else 0
        all_dice.append(d)
        print(f"  {sl_idx:4d}   {bg_pct:5.1f}%  {fg_cls}  dice={d:.3f}")

    print(f"\n  Mean Dice: {np.mean(all_dice):.4f}")
    if np.mean(all_dice) > 0.35:
        print("  ✓ Kaggle model working — ready to use in server")
    else:
        print("  ⚠ Low dice — conversion may have issues")
else:
    print("\n✗ Conversion incomplete — check errors above")
