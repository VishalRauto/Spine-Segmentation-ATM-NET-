# ═══════════════════════════════════════════════════════════
# COMPLETE SELF-CONTAINED TRAINING LOOP
# Paste this as a NEW cell in Kaggle — replaces the broken training cell
# Requires: ti, tm, tr_rare, vi, vm, va_rare, model, device to be defined
# ═══════════════════════════════════════════════════════════
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, time, json, gc
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import defaultdict
from pathlib import Path

OUTPUT_DIR = Path('/kaggle/working')
CKPT_BEST  = OUTPUT_DIR / 'best_model.pth'
CKPT_LAST  = OUTPUT_DIR / 'last_model.pth'

# ── Force writeable arrays ───────────────────────────────
ti_w = np.array(ti, copy=True)
tm_w = np.array(tm, copy=True)
vi_w = np.array(vi, copy=True)
vm_w = np.array(vm, copy=True)

# ── DataLoaders with num_workers=0 ───────────────────────
sampler = WeightedRandomSampler(torch.tensor(tr_rare), len(tr_rare), replacement=True)
tr_dl = DataLoader(DS(ti_w, tm_w, Aug()), batch_size=BATCH_SIZE,
                   sampler=sampler, num_workers=0, pin_memory=False)
va_dl = DataLoader(DS(vi_w, vm_w),        batch_size=BATCH_SIZE,
                   shuffle=False, num_workers=0, pin_memory=False)
print(f'Loaders: {len(tr_dl)} train | {len(va_dl)} val batches')
print(f'num_workers=0 (Kaggle fix)')

# ── Resume from checkpoint if exists ────────────────────
start_ep = 1; best = 0.0
if CKPT_BEST.exists():
    ck = torch.load(str(CKPT_BEST), map_location=device)
    model.load_state_dict(ck['model_state_dict'], strict=False)
    best = ck.get('best_dice', 0.0)
    start_ep = ck.get('epoch', 0) + 1
    print(f'Resumed: ep{ck.get("epoch")} best={best:.4f} → continuing ep{start_ep}')
else:
    print('Starting fresh')

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
scaler    = GradScaler()
no_imp    = 0
t0_total  = time.time()

print(f'\n{"Ep":>4}  {"TrLoss":>8}  {"VaDice":>8}  {"Best":>8}  {"Gap":>6}  {"LR":>8}  {"Sec":>5}')
print('─'*65)

for ep in range(start_ep, EPOCHS+1):
    lr_now = LR_MIN + 0.5*(LR-LR_MIN)*(1+np.cos(np.pi*max(0,ep-WARMUP_EP)/max(EPOCHS-WARMUP_EP,1)))
    if ep <= WARMUP_EP: lr_now = LR * ep / WARMUP_EP
    for pg in optimizer.param_groups: pg['lr'] = lr_now

    # ── Train ──
    model.train(); losses = []; t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    for step, (imgs, msks) in enumerate(tr_dl):
        imgs = imgs.to(device, non_blocking=True)
        msks = msks.to(device, non_blocking=True)
        with autocast():
            outs = model(imgs)
            loss = total_loss(outs, msks) / ACCUM
        scaler.scale(loss).backward()
        if (step+1) % ACCUM == 0 or (step+1) == len(tr_dl):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item() * ACCUM)
    tr_loss = float(np.mean(losses)); ep_sec = time.time() - t0

    # ── Validate ──
    model.eval(); Dc = defaultdict(list)
    with torch.no_grad():
        for imgs, msks in va_dl:
            imgs = imgs.to(device); msks = msks.to(device)
            with autocast():
                p1 = F.softmax(model(imgs), 1)
                p2 = F.softmax(model(torch.flip(imgs,[-1])), 1)
                avg = (p1 + torch.flip(p2,[-1])) / 2
            # fast dice
            pred = avg.argmax(1)
            for c in range(1, NC):
                p = (pred==c).float().view(BATCH_SIZE,-1)
                t = (msks==c).float().view(BATCH_SIZE,-1)
                active = (t.sum(1)>0)|(p.sum(1)>0)
                if not active.any(): continue
                tp = (p*t).sum(1)[active]
                den = (p.sum(1)+t.sum(1))[active]
                Dc[c].extend(((2*tp+1e-6)/(den+1e-6)).cpu().tolist())

    all_v = [v for vs in Dc.values() for v in vs]
    vd = float(np.mean(all_v)) if all_v else 0.0

    # Quick train dice
    with torch.no_grad():
        model.eval()
        imgs_s, msks_s = next(iter(tr_dl))
        with autocast(): out_s = model(imgs_s.to(device))
        pred_s = out_s.argmax(1); sm=1e-6; td_vals=[]
        for c in range(1,NC):
            p=(pred_s==c).float().view(-1); t=(msks_s.to(device)==c).float().view(-1)
            if t.sum()==0: continue
            tp=(p*t).sum(); td_vals.append(float((2*tp+sm)/(p.sum()+t.sum()+sm)))
        td = float(np.mean(td_vals)) if td_vals else 0.0
    gap = td - vd

    # ── Save ──
    if vd > best:
        best = vd; no_imp = 0
        pc = {CN[c]: float(np.mean(v)) for c,v in Dc.items() if v}
        torch.save({'epoch':ep,'model_state_dict':model.state_dict(),
                    'best_dice':best,'per_class_dice':pc,
                    'cfg':{'img_size':IMG_SIZE,'nc':NC}}, CKPT_BEST)
        with open(OUTPUT_DIR/'results.json','w') as f:
            json.dump({'epoch':ep,'best_dice':best,'per_class':pc},f,indent=2)
    else:
        no_imp += 1

    if ep % 5 == 0:
        torch.save({'epoch':ep,'model_state_dict':model.state_dict(),'best_dice':best}, CKPT_LAST)

    flag = '  ★' if vd == best else ''
    print(f'{ep:>4}  {tr_loss:>8.4f}  {vd:>8.4f}  {best:>8.4f}  {gap:>+6.3f}  {lr_now:>8.2e}  {ep_sec:>4.0f}s{flag}')

    if vd >= 0.90: print('\n🎯 Dice ≥ 0.90!'); break
    if vd >= 0.85: print(f'  📈 {vd:.4f} — past 0.85!')
    if no_imp >= PATIENCE: print(f'\nEarly stop ({PATIENCE} epochs)'); break
    gc.collect(); torch.cuda.empty_cache()

t_total = (time.time()-t0_total)/3600
print('─'*65)
print(f'\nDone: {t_total:.2f}h | Best Dice: {best:.4f}')
print(f'Checkpoint: {CKPT_BEST}')
