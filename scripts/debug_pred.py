"""
Deep diagnosis: what is the model actually predicting?
Tests multiple slices at different spine levels.
"""
import sys, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, cv2
from pathlib import Path
from collections import defaultdict

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

device = torch.device("cpu")

# Test BOTH checkpoints
for ckpt_name in ["last_model.pth", "best_model.pth"]:
    ckpt_p = Path(r"c:\project\Spine Segmentation\ATM-Net++\outputs\gpu_run") / ckpt_name
    if not ckpt_p.exists(): continue
    
    model = ResUNet(b=32, nc=NUM_CLASSES, drop=0.25).to(device)
    ckpt = torch.load(str(ckpt_p), map_location=device)
    keys = list(ckpt["model_state_dict"].keys())
    
    print(f"\n{'='*60}")
    print(f"Checkpoint: {ckpt_name}  ep={ckpt.get('epoch')}  dice={ckpt.get('best_dice',0):.4f}")
    print(f"  first key: {keys[0]}")
    
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"  missing={len(missing)} unexpected={len(unexpected)}")
    if missing: print(f"  missing: {missing[:3]}")
    model.eval()

    import SimpleITK as sitk
    vol  = sitk.GetArrayFromImage(sitk.ReadImage(r"c:\project\Spine Segmentation\10159290\images\100_t2.mha")).astype(np.float32)
    n = vol.shape[0]
    
    class_votes = defaultdict(int)
    for sl_idx in [n//4, n//3, n//2, 2*n//3, 3*n//4]:
        sl = vol[sl_idx]
        p1, p99 = np.percentile(sl, [0.5, 99.5])
        img_n = np.clip((sl - p1) / (p99 - p1 + 1e-8), 0, 1).astype(np.float32)
        img_r = cv2.resize(img_n, (192, 192), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(img_r[None, None]).float()
        
        with torch.no_grad():
            out = model(t)
            probs = F.softmax(out, 1).squeeze()
            pred = out.argmax(1).squeeze().numpy()
        
        top_class = int(probs.max(0)[1].flatten()[probs.max(0)[0].argmax()])
        dominant = np.bincount(pred.flatten()).argmax()
        for c in np.unique(pred):
            class_votes[c] += int((pred==c).sum())
        
        # Check softmax confidence — if one class >> all others, it's degenerate
        max_softmax = probs.max().item()
        entropy = -(probs * (probs.clamp(min=1e-8)).log()).sum(0).mean().item()
        unique_classes = np.unique(pred)
        
        print(f"\n  Slice {sl_idx}: dominant={CLASS_NAMES[dominant]}({dominant}) "
              f"unique={len(unique_classes)} max_softmax={max_softmax:.3f} entropy={entropy:.3f}")
        print(f"    Classes: {[CLASS_NAMES[c] for c in unique_classes]}")
        
        # Print top 3 class logits
        mean_logits = out.squeeze().mean(-1).mean(-1)
        top3 = mean_logits.topk(5)
        print(f"    Top5 mean logits: {[(CLASS_NAMES[i.item()], f'{v.item():.3f}') for v,i in zip(top3.values, top3.indices)]}")
    
    print(f"\n  Total pixel votes across all slices:")
    total_px = sum(class_votes.values())
    for c in sorted(class_votes, key=lambda x: -class_votes[x])[:8]:
        pct = class_votes[c]/total_px*100
        print(f"    {CLASS_NAMES[c]:6s}({c:2d}): {pct:5.1f}%")
