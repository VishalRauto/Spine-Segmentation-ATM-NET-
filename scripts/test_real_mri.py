"""Test with a real MRI slice from the SPIDER dataset."""
import sys, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, cv2
from pathlib import Path

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

SPIDER_TO_ATMNET = {**{i: i for i in range(1, 9)}, 100: 9, **{201+i: 10+i for i in range(8)}}
CLASS_NAMES = {0:"bg",1:"V1",2:"V2",3:"V3",4:"V4",5:"V5",6:"V6",7:"V7",8:"V8",9:"Sac",
               10:"I1",11:"I2",12:"I3",13:"I4",14:"I5",15:"I6",16:"I7",17:"I8",18:"Canal"}

# Load model
device = torch.device("cpu")
model  = ResUNet(b=32, nc=NUM_CLASSES, drop=0.25).to(device)
ckpt_p = Path(r"c:\project\Spine Segmentation\ATM-Net++\outputs\gpu_run\last_model.pth")
ckpt   = torch.load(str(ckpt_p), map_location=device)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()
print(f"Model loaded: epoch={ckpt.get('epoch','?')} dice={ckpt.get('best_dice',0):.4f}")

# Load a real T2 MRI
import SimpleITK as sitk
mha = Path(r"c:\project\Spine Segmentation\10159290\images\100_t2.mha")
mha_mask = Path(r"c:\project\Spine Segmentation\10159290\masks\100_t2.mha")
vol  = sitk.GetArrayFromImage(sitk.ReadImage(str(mha))).astype(np.float32)
mask_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(mha_mask))).astype(np.int32)
n = vol.shape[0]
# Use mid-spine slice
sl_idx = n // 2
sl   = vol[sl_idx]
gt   = mask_vol[sl_idx]

# Preprocess
p1, p99 = np.percentile(sl, [0.5, 99.5])
img_n = np.clip((sl - p1) / (p99 - p1 + 1e-8), 0, 1).astype(np.float32)
img_r = cv2.resize(img_n, (192, 192), interpolation=cv2.INTER_LINEAR)
t = torch.from_numpy(img_r[None, None]).float()

with torch.no_grad():
    # TTA: flip + average
    p1 = F.softmax(model(t), 1)
    p2 = F.softmax(model(torch.flip(t, [-1])), 1)
    p2 = torch.flip(p2, [-1])
    avg = (p1 + p2) / 2
    pred = avg.argmax(1).squeeze().numpy()

unique = np.unique(pred)
print(f"\nSlice {sl_idx}/{n}")
print(f"Predicted classes: {[CLASS_NAMES[c] for c in unique]}")
print(f"\nPixel counts per class:")
total = pred.size
for c in unique:
    cnt = (pred == c).sum()
    pct = cnt / total * 100
    print(f"  {CLASS_NAMES[c]:6s} ({c:2d}): {cnt:6d} px  {pct:5.1f}%")

# Compute dice vs ground truth
def remap(m):
    out = np.zeros_like(m, dtype=np.int32)
    for s,d in SPIDER_TO_ATMNET.items(): out[m==s]=d
    return out

gt_r = remap(cv2.resize(gt.astype(np.float32), (192,192), interpolation=cv2.INTER_NEAREST).astype(np.int32))
gt_classes = np.unique(gt_r)
print(f"\nGround truth classes: {[CLASS_NAMES[c] for c in gt_classes if c < NUM_CLASSES]}")
print(f"\nDice per class (pred vs GT):")
sm = 1e-6
all_dice = []
for c in range(1, NUM_CLASSES):
    p = (pred == c).astype(float).ravel()
    g = (gt_r  == c).astype(float).ravel()
    if g.sum() == 0 and p.sum() == 0: continue
    tp = (p*g).sum()
    dice = (2*tp + sm) / (p.sum() + g.sum() + sm)
    all_dice.append(dice)
    flag = "  ★" if dice >= 0.8 else "  ✓" if dice >= 0.6 else ""
    print(f"  {CLASS_NAMES[c]:6s} ({c:2d}): {dice:.4f}{flag}")

if all_dice:
    print(f"\n  Mean Dice on this slice: {np.mean(all_dice):.4f}")
    if len(unique) > 3:
        print("  ✓ Model is producing multi-class output — server should show correct colors")
    else:
        print("  ⚠ Still mostly single class — model needs more training")
