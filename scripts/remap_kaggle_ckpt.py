"""
Remap Kaggle checkpoint (e1.c.0.weight keys) → local architecture (e1.conv.0.weight keys).

The Kaggle model was trained with a slightly different Enc class:
  Kaggle:  self.c = nn.Sequential(...)    → keys: e1.c.0.weight
  Local:   self.conv = nn.Sequential(...) → keys: e1.conv.0.weight

Both have identical layer structure, just different attribute names.
We remap key names and save a compatible checkpoint.

Result: outputs/gpu_run/best_model_fixed.pth
  - Same weights as Kaggle epoch 48 (Dice 0.6529)
  - Compatible with local server architecture
"""
import torch
from pathlib import Path
from collections import OrderedDict

KAGGLE = Path(r"c:\project\Spine Segmentation\ATM-Net++\outputs\gpu_run\best_model.pth")
OUT    = Path(r"c:\project\Spine Segmentation\ATM-Net++\outputs\gpu_run\best_model_fixed.pth")
LOCAL  = Path(r"c:\project\Spine Segmentation\ATM-Net++\outputs\gpu_run\last_model.pth")

kaggle_ckpt = torch.load(str(KAGGLE), map_location="cpu")
local_ckpt  = torch.load(str(LOCAL),  map_location="cpu")

kaggle_sd = kaggle_ckpt["model_state_dict"]
local_sd  = local_ckpt["model_state_dict"]

print("Kaggle keys sample:", list(kaggle_sd.keys())[:6])
print("Local  keys sample:", list(local_sd.keys())[:6])

# Build key mapping: kaggle → local
# Pattern: 'e1.c.0.weight'      → 'e1.conv.0.weight'
#           'e1.c.1.weight'      → 'e1.conv.1.weight'  etc.
#           'e1.res.net.0.weight'→ same (res block unchanged)
#           'e1.res.ca.fc.1.weight' → same
# Also: Kaggle Enc has no 'drop' layer (it's Identity in local when drop=0)
# Kaggle also has no aux head, no ds3/ds2 in some versions.

import re

def remap_key(k):
    # Replace '.c.' with '.conv.' in encoder/decoder block names
    # e.g. e1.c.0.weight → e1.conv.0.weight
    #      d4.c.0.weight → d4.conv.0.weight
    #      bn.0.c.0.weight → bn.0.conv.0.weight
    k2 = re.sub(r'\.c\.(\d+)', r'.conv.\1', k)
    return k2

new_sd = OrderedDict()
mapped  = 0
skipped = 0
unmapped = []

for k, v in kaggle_sd.items():
    k2 = remap_key(k)
    if k2 in local_sd:
        if local_sd[k2].shape == v.shape:
            new_sd[k2] = v
            mapped += 1
        else:
            print(f"  Shape mismatch: {k2}: kaggle={tuple(v.shape)} local={tuple(local_sd[k2].shape)}")
            new_sd[k2] = local_sd[k2]  # keep local
            skipped += 1
    else:
        unmapped.append(k2)

# Fill any remaining local keys not covered by kaggle (aux, ds3, ds2, drop)
for k, v in local_sd.items():
    if k not in new_sd:
        new_sd[k] = v
        print(f"  Filled from local: {k}")

print(f"\nMapped  : {mapped}")
print(f"Skipped : {skipped}")
print(f"Unmapped: {len(unmapped)}")
if unmapped:
    print(f"  First 5: {unmapped[:5]}")

# Verify all local keys are present
missing = [k for k in local_sd if k not in new_sd]
extra   = [k for k in new_sd if k not in local_sd]
print(f"\nFinal check:")
print(f"  Missing from local schema: {len(missing)}")
print(f"  Extra keys: {len(extra)}")
if missing: print(f"  Missing: {missing[:5]}")

# Save
torch.save({
    "epoch"            : kaggle_ckpt.get("epoch", 48),
    "best_dice"        : kaggle_ckpt.get("best_dice", 0.6529),
    "model_state_dict" : new_sd,
    "per_class_dice"   : kaggle_ckpt.get("per_class_dice", {}),
    "cfg"              : {"sz": 256, "nc": 19, "source": "kaggle_remapped"},
}, str(OUT))
print(f"\n✓ Saved to: {OUT}")
print(f"  epoch={kaggle_ckpt.get('epoch',48)}  dice={kaggle_ckpt.get('best_dice',0.6529):.4f}")

# Quick load test
print("\nVerifying load...")
import torch.nn as nn, torch.nn.functional as F
NUM_CLASSES = 19

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
        self.res=RB(co)
        self.drop=nn.Dropout2d(drop) if drop>0 else nn.Identity()
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
        e1=self.e1(x); e2=self.e2(self.pool(e1)); e3=self.e3(self.pool(e2)); e4=self.e4(self.pool(e3))
        d=self.bn(self.pool(e4))
        d=self.d4(torch.cat([self.u4(d),e4],1))
        d=self.d3(torch.cat([self.u3(d),e3],1))
        d=self.d2(torch.cat([self.u2(d),e2],1))
        d=self.d1(torch.cat([self.u1(d),e1],1))
        return self.out(d)

model = ResUNet(b=32, nc=NUM_CLASSES, drop=0.25)
ck2 = torch.load(str(OUT), map_location="cpu")
m, u = model.load_state_dict(ck2["model_state_dict"], strict=False)
print(f"  missing={len(m)}  unexpected={len(u)}")

# Run on a real slice
import numpy as np, cv2, SimpleITK as sitk
vol = sitk.GetArrayFromImage(sitk.ReadImage(r"c:\project\Spine Segmentation\10159290\images\100_t2.mha")).astype(np.float32)
n = vol.shape[0]
sl = vol[n//2]
p1,p99 = np.percentile(sl,[0.5,99.5])
img_n = np.clip((sl-p1)/(p99-p1+1e-8),0,1).astype(np.float32)
img_r = cv2.resize(img_n,(192,192),interpolation=cv2.INTER_LINEAR)
t = torch.from_numpy(img_r[None,None]).float()
model.eval()
with torch.no_grad():
    out = model(t)
    pred = out.argmax(1).squeeze().numpy()

CLASS_NAMES = {0:"bg",1:"V1",2:"V2",3:"V3",4:"V4",5:"V5",6:"V6",7:"V7",8:"V8",9:"Sac",
               10:"I1",11:"I2",12:"I3",13:"I4",14:"I5",15:"I6",16:"I7",17:"I8",18:"Canal"}
unique = np.unique(pred)
print(f"\nPrediction on real MRI (mid slice):")
print(f"  Classes: {[CLASS_NAMES[c] for c in unique]}")
total = pred.size
for c in unique:
    pct = (pred==c).sum()/total*100
    print(f"    {CLASS_NAMES[c]:6s}: {pct:.1f}%")

bg_pct = (pred==0).sum()/total*100
if bg_pct < 80:
    print(f"\n✓ GOOD: background={bg_pct:.1f}%, model producing proper foreground segmentation")
else:
    print(f"\n⚠ High background: {bg_pct:.1f}% — remapped weights may not align properly")
    print("  Check top5 logits:")
    mean_logits = out.squeeze().mean(-1).mean(-1)
    top5 = mean_logits.topk(5)
    for v,i in zip(top5.values, top5.indices):
        print(f"    {CLASS_NAMES[i.item()]}: {v.item():.4f}")
