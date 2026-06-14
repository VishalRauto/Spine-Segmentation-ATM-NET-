import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, cv2, SimpleITK as sitk
from pathlib import Path

NC = 19
CLASS_NAMES = {0:'bg',1:'V1',2:'V2',3:'V3',4:'V4',5:'V5',6:'V6',7:'V7',8:'V8',9:'Sac',
               10:'I1',11:'I2',12:'I3',13:'I4',14:'I5',15:'I6',16:'I7',17:'I8',18:'Canal'}
SPIDER_TO_ATMNET = {**{i:i for i in range(1,9)}, 100:9, **{201+i:10+i for i in range(8)}}

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
        self.bn=nn.Sequential(Enc(b*8,b*16,drop),nn.Dropout2d(drop)); self.pool=nn.MaxPool2d(2)
        self.u4=nn.ConvTranspose2d(b*16,b*8,2,2); self.d4=Enc(b*16,b*8,drop*0.4)
        self.u3=nn.ConvTranspose2d(b*8,b*4,2,2);  self.d3=Enc(b*8,b*4,drop*0.2)
        self.u2=nn.ConvTranspose2d(b*4,b*2,2,2);  self.d2=Enc(b*4,b*2)
        self.u1=nn.ConvTranspose2d(b*2,b,2,2);    self.d1=Enc(b*2,b)
        self.ds3=nn.Conv2d(b*4,nc,1); self.ds2=nn.Conv2d(b*2,nc,1); self.out=nn.Conv2d(b,nc,1)
        self.aux=nn.Sequential(nn.Conv2d(b,b,3,1,1,bias=False),nn.BatchNorm2d(b),nn.ReLU(True),nn.Conv2d(b,nc,1))
    def forward(self,x):
        e1=self.e1(x); e2=self.e2(self.pool(e1)); e3=self.e3(self.pool(e2)); e4=self.e4(self.pool(e3))
        d=self.bn(self.pool(e4))
        d=self.d4(torch.cat([self.u4(d),e4],1)); d=self.d3(torch.cat([self.u3(d),e3],1))
        d=self.d2(torch.cat([self.u2(d),e2],1)); d=self.d1(torch.cat([self.u1(d),e1],1))
        return self.out(d)

model = ResUNet(b=32,nc=NC,drop=0.25)
ck = torch.load(r'c:\project\Spine Segmentation\ATM-Net++\outputs\gpu_run\kaggle_converted.pth', map_location='cpu')
model.load_state_dict(ck['model_state_dict'], strict=True)
model.eval()
print(f"Loaded ep={ck.get('epoch')} dice={ck.get('best_dice',0):.4f} — testing at 192 vs 256\n")

vol  = sitk.GetArrayFromImage(sitk.ReadImage(r'c:\project\Spine Segmentation\10159290\images\100_t2.mha')).astype(np.float32)
mskv = sitk.GetArrayFromImage(sitk.ReadImage(r'c:\project\Spine Segmentation\10159290\masks\100_t2.mha')).astype(np.int32)
n = vol.shape[0]

def remap(m):
    out = np.zeros_like(m, dtype=np.int32)
    for s,d in SPIDER_TO_ATMNET.items(): out[m==s]=d
    return out

print(f"{'Slice':>5}  {'@192':>6}  {'@256':>6}  FG classes @256")
print("-"*55)
all_192, all_256 = [], []
for sl_idx in [n//5, n//4, n//3, n//2, 2*n//3]:
    sl  = vol[sl_idx]; msk = mskv[sl_idx]
    p1,p99 = np.percentile(sl,[0.5,99.5])
    img_n = np.clip((sl-p1)/(p99-p1+1e-8),0,1).astype(np.float32)
    sm = 1e-6
    results = {}
    for size in [192, 256]:
        img_r = cv2.resize(img_n,(size,size),interpolation=cv2.INTER_LINEAR)
        gt_r  = remap(cv2.resize(msk.astype(np.float32),(size,size),
                                 interpolation=cv2.INTER_NEAREST).astype(np.int32))
        t = torch.from_numpy(img_r[None,None]).float()
        with torch.no_grad():
            pr  = F.softmax(model(t),1)
            pr2 = F.softmax(model(torch.flip(t,[-1])),1)
            avg = ((pr+torch.flip(pr2,[-1]))/2).squeeze(0).numpy()
        pred = avg.argmax(0).astype(np.int32)
        dices = []
        for c in range(1,19):
            g=(gt_r==c).astype(float).ravel()
            if g.sum()==0: continue
            p=(pred==c).astype(float).ravel()
            tp=(p*g).sum()
            dices.append((2*tp+sm)/(p.sum()+g.sum()+sm))
        fg_cls = [CLASS_NAMES[c] for c in np.unique(pred) if c>0]
        results[size] = (np.mean(dices) if dices else 0, fg_cls)
    all_192.append(results[192][0]); all_256.append(results[256][0])
    print(f"  {sl_idx:4d}  {results[192][0]:>6.3f}  {results[256][0]:>6.3f}  {results[256][1]}")

print(f"\nMean @192: {np.mean(all_192):.4f}")
print(f"Mean @256: {np.mean(all_256):.4f}")
if np.mean(all_256) > np.mean(all_192):
    print(f"\n=> USE 256x256 for Kaggle model inference (+{np.mean(all_256)-np.mean(all_192):.3f} Dice)")
else:
    print(f"\n=> 192x192 is fine")
