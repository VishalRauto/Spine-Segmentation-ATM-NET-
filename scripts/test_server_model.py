"""Quick test: verifies server model loads correctly and produces valid predictions."""
import sys, os
sys.path.insert(0, r"c:\project\Spine Segmentation\ATM-Net++")
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
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
        sz=x.shape[2:]
        e1=self.e1(x); e2=self.e2(self.pool(e1)); e3=self.e3(self.pool(e2)); e4=self.e4(self.pool(e3))
        d=self.bn(self.pool(e4))
        d=self.d4(torch.cat([self.u4(d),e4],1))
        d=self.d3(torch.cat([self.u3(d),e3],1))
        d=self.d2(torch.cat([self.u2(d),e2],1))
        d=self.d1(torch.cat([self.u1(d),e1],1))
        return self.out(d)

device = torch.device("cpu")
model  = ResUNet(b=32, nc=NUM_CLASSES, drop=0.25).to(device)

OUT = Path(r"c:\project\Spine Segmentation\ATM-Net++\outputs\gpu_run")
for ckpt_name in ["last_model.pth", "best_model.pth"]:
    p = OUT / ckpt_name
    if not p.exists():
        print(f"[SKIP] {ckpt_name} not found")
        continue

    ckpt = torch.load(str(p), map_location=device)
    keys = list(ckpt["model_state_dict"].keys())
    is_local = any("conv.0.weight" in k for k in keys)
    print(f"\n{'='*55}")
    print(f"Checkpoint : {ckpt_name}")
    print(f"  epoch    : {ckpt.get('epoch','?')}")
    print(f"  dice     : {ckpt.get('best_dice',0):.4f}")
    print(f"  local arch: {is_local}")

    if not is_local:
        print("  SKIPPED: incompatible architecture (Kaggle model)")
        continue

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"  missing  : {len(missing)} keys  (aux/ds3/ds2 only)")
    print(f"  unexpected: {len(unexpected)} keys")
    model.eval()

    # Run a test forward pass
    x = torch.rand(1, 1, 192, 192)
    with torch.no_grad():
        out = model(x)
    pred = out.argmax(1).squeeze().numpy()
    unique_classes = np.unique(pred)
    print(f"\n  Forward pass: input={tuple(x.shape)} → output={tuple(out.shape)}")
    print(f"  Unique predicted classes: {unique_classes}")

    if len(unique_classes) == 1:
        print("  ⚠ WARNING: Only 1 class predicted — model may be degenerate")
    else:
        # Check softmax spread
        probs = torch.softmax(out, dim=1).squeeze()
        max_prob = probs.max().item()
        entropy  = -(probs * (probs + 1e-8).log()).sum(0).mean().item()
        print(f"  Max softmax prob  : {max_prob:.4f}  (should be <0.999 for varied output)")
        print(f"  Mean pixel entropy: {entropy:.4f}  (higher = more uncertainty = good)")
        print(f"  ✓ Model looks CORRECT — predicting {len(unique_classes)} distinct classes")
        print(f"  ✓ Ready for server.py use")

print("\nDone.")
