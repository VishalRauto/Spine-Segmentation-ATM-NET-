import torch, time
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Test multiple configs to find best speed/quality tradeoff
configs = [
    {"sz":64,  "b":16, "bs":32, "n":1032},
    {"sz":96,  "b":24, "bs":16, "n":1032},
    {"sz":128, "b":16, "bs":32, "n":1032},
]

def cb(ci,co):
    return nn.Sequential(
        nn.Conv2d(ci,co,3,1,1,bias=False),nn.BatchNorm2d(co),nn.ReLU(True),
        nn.Conv2d(co,co,3,1,1,bias=False),nn.BatchNorm2d(co),nn.ReLU(True))

class Net(nn.Module):
    def __init__(self,b):
        super().__init__()
        self.e1=cb(1,b);self.e2=cb(b,b*2);self.e3=cb(b*2,b*4)
        self.bn=cb(b*4,b*8);self.pool=nn.MaxPool2d(2)
        self.u3=nn.ConvTranspose2d(b*8,b*4,2,2);self.d3=cb(b*8,b*4)
        self.u2=nn.ConvTranspose2d(b*4,b*2,2,2);self.d2=cb(b*4,b*2)
        self.u1=nn.ConvTranspose2d(b*2,b,2,2);self.d1=cb(b*2,b)
        self.out=nn.Conv2d(b,19,1)
    def forward(self,x):
        e1=self.e1(x);e2=self.e2(self.pool(e1));e3=self.e3(self.pool(e2))
        d=self.bn(self.pool(e3))
        d=self.d3(torch.cat([self.u3(d),e3],1))
        d=self.d2(torch.cat([self.u2(d),e2],1))
        d=self.d1(torch.cat([self.u1(d),e1],1))
        return self.out(d)

for cfg in configs:
    sz=cfg["sz"]; b=cfg["b"]; bs=cfg["bs"]; n=cfg["n"]
    imgs=torch.randn(n,1,sz,sz); masks=torch.randint(0,19,(n,sz,sz))
    dl=DataLoader(TensorDataset(imgs,masks),batch_size=bs,shuffle=True,num_workers=0)
    model=Net(b); optim=torch.optim.Adam(model.parameters())
    np_=sum(p.numel() for p in model.parameters())
    t0=time.time()
    for i,(im,mk) in enumerate(dl):
        optim.zero_grad()
        F.cross_entropy(model(im),mk).backward()
        optim.step()
        if i==3: break
    t1=time.time()
    per_b=(t1-t0)/4; nb=len(dl); ep=per_b*nb
    print(f"sz={sz:<4} b={b:<3} bs={bs:<3} params={np_/1e6:.2f}M  "
          f"ep={ep:.0f}s({ep/60:.1f}m)  50ep={ep*50/60:.0f}min")
