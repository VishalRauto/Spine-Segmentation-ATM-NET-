"""Test background-bias correction on real MRI slices."""
import sys, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, cv2, SimpleITK as sitk
from pathlib import Path

NUM_CLASSES = 19
CLASS_NAMES = {0:"bg",1:"V1",2:"V2",3:"V3",4:"V4",5:"V5",6:"V6",7:"V7",8:"V8",9:"Sac",
               10:"I1",11:"I2",12:"I3",13:"I4",14:"I5",15:"I6",16:"I7",17:"I8",18:"Canal"}
SPIDER_TO_ATMNET = {**{i: i for i in range(1,9)}, 100:9, **{201+i:10+i for i in range(8)}}

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

def bg_correction(avg_np):
    """Suppress background bias. avg_np: [C,H,W] softmax probabilities."""
    bg_prob  = avg_np[0].copy()
    fg_probs = avg_np[1:]
    fg_max   = fg_probs.max(0)
    bg_bias  = np.clip(bg_prob - fg_max - 0.02, 0, None)
    avg_np   = avg_np.copy()
    avg_np[0] = bg_prob - bg_bias
    avg_np   = avg_np / (avg_np.sum(0, keepdims=True) + 1e-8)
    return avg_np

def remap(m):
    out = np.zeros_like(m, dtype=np.int32)
    for s,d in SPIDER_TO_ATMNET.items(): out[m==s]=d
    return out

device = torch.device("cpu")
OUT = Path(r"c:\project\Spine Segmentation\ATM-Net++\outputs\gpu_run")

for ckpt_name in ["last_model.pth"]:
    p = OUT / ckpt_name
    if not p.exists(): continue
    model = ResUNet(b=32, nc=NUM_CLASSES, drop=0.25)
    ckpt = torch.load(str(p), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"Checkpoint: {ckpt_name}  ep={ckpt.get('epoch')}  dice={ckpt.get('best_dice',0):.4f}\n")

    vol  = sitk.GetArrayFromImage(sitk.ReadImage(r"c:\project\Spine Segmentation\10159290\images\100_t2.mha")).astype(np.float32)
    mskv = sitk.GetArrayFromImage(sitk.ReadImage(r"c:\project\Spine Segmentation\10159290\masks\100_t2.mha")).astype(np.int32)
    n    = vol.shape[0]

    all_dice_raw, all_dice_fix = [], []

    for sl_idx in [n//5, n//4, n//3, n//2, 2*n//3, 3*n//4]:
        sl  = vol[sl_idx]
        msk = mskv[sl_idx]
        p1,p99 = np.percentile(sl,[0.5,99.5])
        img_n = np.clip((sl-p1)/(p99-p1+1e-8),0,1).astype(np.float32)
        img_r = cv2.resize(img_n,(192,192),interpolation=cv2.INTER_LINEAR)
        gt_r  = remap(cv2.resize(msk.astype(np.float32),(192,192),interpolation=cv2.INTER_NEAREST).astype(np.int32))
        t = torch.from_numpy(img_r[None,None]).float()

        with torch.no_grad():
            pr  = F.softmax(model(t),1)
            pr2 = F.softmax(model(torch.flip(t,[-1])),1)
            pr2 = torch.flip(pr2,[-1])
            avg = ((pr+pr2)/2).squeeze(0).numpy()

        pred_raw = avg.argmax(0).astype(np.int32)
        avg_fix  = bg_correction(avg)
        pred_fix = avg_fix.argmax(0).astype(np.int32)

        # Dice for both
        sm = 1e-6
        dice_raw, dice_fix = [], []
        for c in range(1,NUM_CLASSES):
            g = (gt_r==c).astype(float).ravel()
            if g.sum() == 0: continue
            pr_r = (pred_raw==c).astype(float).ravel()
            pr_f = (pred_fix==c).astype(float).ravel()
            tp_r = (pr_r*g).sum(); tp_f = (pr_f*g).sum()
            d_r = (2*tp_r+sm)/(pr_r.sum()+g.sum()+sm)
            d_f = (2*tp_f+sm)/(pr_f.sum()+g.sum()+sm)
            dice_raw.append(d_r); dice_fix.append(d_f)

        raw_mean = np.mean(dice_raw) if dice_raw else 0
        fix_mean = np.mean(dice_fix) if dice_fix else 0
        all_dice_raw.append(raw_mean); all_dice_fix.append(fix_mean)
        bg_raw_pct = (pred_raw==0).sum()/pred_raw.size*100
        bg_fix_pct = (pred_fix==0).sum()/pred_fix.size*100
        classes_fix = [CLASS_NAMES[c] for c in np.unique(pred_fix) if c > 0]
        print(f"  Slice {sl_idx:3d}: RAW dice={raw_mean:.3f} bg={bg_raw_pct:.0f}%  |  "
              f"FIX dice={fix_mean:.3f} bg={bg_fix_pct:.0f}%  classes={classes_fix}")

    print(f"\n  Mean Dice RAW: {np.mean(all_dice_raw):.4f}")
    print(f"  Mean Dice FIX: {np.mean(all_dice_fix):.4f}")
    improvement = np.mean(all_dice_fix) - np.mean(all_dice_raw)
    print(f"  Improvement  : +{improvement:.4f}")
    if np.mean(all_dice_fix) > 0.3:
        print("\n  ✓ Background correction working — server will show proper segmentation")
    else:
        print("\n  ⚠ Still low Dice — model needs more training")
