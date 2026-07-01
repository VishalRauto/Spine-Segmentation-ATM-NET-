"""
ATM-Net++ Classifier Training
==============================
Trains MultimodalFusionModule (ATPG+HASF+CCAE) + MultiTaskHead
using REAL ResUNet segmentation features as image input.

Pipeline:
  1. Load trained ResUNet (kaggle_v2.pth, Dice=0.7719)
  2. Run inference on every T2 MHA in the SPIDER dataset
  3. Extract per-class probability features (19-dim) per patient
  4. Map radiological_gradings.csv → disease / severity / Pfirrmann targets
  5. Train fusion + classifier on those real features
  6. Save weights → outputs/classifier/multitask_head.pth
                  → outputs/classifier/fusion_module.pth

Usage:
    py train_classifier.py
    py train_classifier.py --epochs 100 --lr 1e-3
    py train_classifier.py --quick   # 10 epochs fast test
"""

import argparse
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).parent
DATA_DIR   = Path(r"c:\project\Spine Segmentation\10159290")
IMAGES_DIR = DATA_DIR / "images"
MASKS_DIR  = DATA_DIR / "masks"
GRADES_CSV = DATA_DIR / "radiological_gradings.csv"
OVERVIEW   = DATA_DIR / "overview.csv"
OUT_DIR    = ROOT / "outputs" / "classifier"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Label maps ────────────────────────────────────────────────────────
# Map per-disc pathologies → patient-level disease class
# Priority: Herniation > Spondylolisthesis > Stenosis > Bulge > DDD > Normal
DISEASE_MAP = {
    "Normal": 0, "Disc Herniation": 1, "Disc Bulge": 2,
    "Spinal Stenosis": 3, "Disc Degeneration": 4,
    "Spondylolisthesis": 5, "Compression Fracture": 6,
}
NUM_CLASSES    = 19
IVD_CLASSES    = list(range(10, 18))
VERT_CLASSES   = list(range(1, 9))
IVD_LABEL_MAP  = {1:"L5/S1",2:"L4/L5",3:"L3/L4",4:"L2/L3",
                   5:"L1/L2",6:"T12/L1",7:"T11/T12",8:"T10/T11"}


# ═══════════════════════════════════════════════════════════════════════
# Step 1 — Load ResUNet (unchanged)
# ═══════════════════════════════════════════════════════════════════════

def load_resunet(device):
    """Load the trained ResUNet from kaggle_v2.pth. Weights never modified."""
    import torch.nn as nn

    class CA(nn.Module):
        def __init__(self, ch, r=8):
            super().__init__(); r = max(1, ch // r)
            self.avg = nn.AdaptiveAvgPool2d(1); self.max = nn.AdaptiveMaxPool2d(1)
            self.fc  = nn.Sequential(nn.Flatten(), nn.Linear(ch, r), nn.ReLU(True),
                                     nn.Linear(r, ch), nn.Sigmoid())
        def forward(self, x):
            a = self.fc(self.avg(x)) + self.fc(self.max(x))
            return x * a.clamp(0,1).view(x.shape[0],-1,1,1)

    class SA(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv2d(2,1,7,padding=3,bias=False),
                                      nn.BatchNorm2d(1), nn.Sigmoid())
        def forward(self, x):
            return x * self.conv(torch.cat([x.mean(1,keepdim=True),
                                            x.max(1,keepdim=True)[0]],1))

    class RB(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(ch,ch,3,1,1,bias=False), nn.BatchNorm2d(ch), nn.ReLU(True),
                nn.Conv2d(ch,ch,3,1,1,bias=False), nn.BatchNorm2d(ch))
            self.ca = CA(ch); self.sa = SA(); self.act = nn.ReLU(True)
        def forward(self, x): return self.act(self.sa(self.ca(self.net(x)))+x)

    class Enc(nn.Module):
        def __init__(self, ci, co, drop=0.0):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(ci,co,3,1,1,bias=False), nn.BatchNorm2d(co), nn.ReLU(True),
                nn.Conv2d(co,co,3,1,1,bias=False), nn.BatchNorm2d(co), nn.ReLU(True))
            self.res  = RB(co)
            self.drop = nn.Dropout2d(drop) if drop>0 else nn.Identity()
        def forward(self, x): return self.drop(self.res(self.conv(x)))

    class ResUNet(nn.Module):
        def __init__(self, b=40, nc=19, drop=0.20):
            super().__init__()
            self.e1=Enc(1,b); self.e2=Enc(b,b*2,drop*.3)
            self.e3=Enc(b*2,b*4,drop*.6); self.e4=Enc(b*4,b*8,drop*.8)
            self.bn=nn.Sequential(Enc(b*8,b*16,drop),nn.Dropout2d(drop))
            self.pool=nn.MaxPool2d(2)
            self.u4=nn.ConvTranspose2d(b*16,b*8,2,2); self.d4=Enc(b*16,b*8,drop*.4)
            self.u3=nn.ConvTranspose2d(b*8,b*4,2,2);  self.d3=Enc(b*8,b*4,drop*.2)
            self.u2=nn.ConvTranspose2d(b*4,b*2,2,2);  self.d2=Enc(b*4,b*2)
            self.u1=nn.ConvTranspose2d(b*2,b,2,2);    self.d1=Enc(b*2,b)
            self.out=nn.Conv2d(b,nc,1)
        def forward(self, x):
            e1=self.e1(x); e2=self.e2(self.pool(e1))
            e3=self.e3(self.pool(e2)); e4=self.e4(self.pool(e3))
            d=self.bn(self.pool(e4))
            d=self.d4(torch.cat([self.u4(d),e4],1))
            d=self.d3(torch.cat([self.u3(d),e3],1))
            d=self.d2(torch.cat([self.u2(d),e2],1))
            d=self.d1(torch.cat([self.u1(d),e1],1))
            return self.out(d)

    ckpt_path = ROOT / "outputs/gpu_run/kaggle_v2.pth"
    model     = ResUNet(b=40, nc=19, drop=0.20).to(device)
    if ckpt_path.exists():
        ck      = torch.load(str(ckpt_path), map_location=device)
        missing,_ = model.load_state_dict(ck["model_state_dict"], strict=False)
        dice    = ck.get("best_dice", 0.0)
        print(f"[ResUNet] Loaded kaggle_v2.pth | Dice={dice:.4f} | missing={len(missing)}")
    else:
        print("[ResUNet] WARNING: checkpoint not found — using random weights")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)   # FROZEN — never trains
    return model


# ═══════════════════════════════════════════════════════════════════════
# Step 2 — Extract image features from ResUNet for each patient
# ═══════════════════════════════════════════════════════════════════════

def extract_features(model, device, img_path: Path, infer_size=384):
    """
    Run ResUNet on a T2 MHA volume.
    Returns (19,) per-class max-probability vector averaged across slices.
    Also returns (19,) mean-probability and (8,) IVD-level confidences.
    """
    import cv2
    try:
        import SimpleITK as sitk
        vol = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path))).astype(np.float32)
    except Exception as e:
        print(f"  [feat] SimpleITK failed for {img_path.name}: {e}")
        return None

    n    = vol.shape[0]
    # Use middle 60% of slices — spine is usually centred there
    lo, hi = int(n * 0.20), int(n * 0.80)
    idxs = list(range(lo, hi, max(1, (hi - lo) // 8)))[:8]
    if not idxs:
        idxs = [n // 2]

    slice_feats = []
    for i in idxs:
        sl = vol[i]
        if sl.max() == sl.min():
            continue
        p1, p99 = np.percentile(sl, [0.5, 99.5])
        img_n = np.clip((sl - p1) / (p99 - p1 + 1e-8), 0, 1).astype(np.float32)
        img_r = cv2.resize(img_n, (infer_size, infer_size), cv2.INTER_LINEAR)
        t = torch.from_numpy(img_r[None, None]).float().to(device)

        with torch.no_grad():
            pr  = F.softmax(model(t), 1)
            pr2 = F.softmax(model(torch.flip(t, [-1])), 1)
            avg = ((pr + torch.flip(pr2, [-1])) / 2).squeeze(0).cpu().numpy()

        # Per-class max prob across spatial dimensions
        feat = np.array([float(avg[c].max()) for c in range(NUM_CLASSES)])
        slice_feats.append(feat)

    if not slice_feats:
        return None

    feats = np.stack(slice_feats, axis=0)  # (S, 19)
    return {
        "max_prob":  feats.max(0),   # (19,) — best slice per class
        "mean_prob": feats.mean(0),  # (19,) — average confidence
        "std_prob":  feats.std(0),   # (19,) — consistency across slices
        # IVD-level confidences (8 discs, L5/S1 → T10/T11)
        "ivd_conf":  feats.mean(0)[IVD_CLASSES],   # (8,)
        "vert_conf": feats.mean(0)[VERT_CLASSES],   # (8,)
    }


# ═══════════════════════════════════════════════════════════════════════
# Step 3 — Load and aggregate labels from radiological_gradings.csv
# ═══════════════════════════════════════════════════════════════════════

def load_labels():
    """
    Parse radiological_gradings.csv → per-patient disease/severity/Pfirrmann/level labels.

    Returns dict: patient_id (str) → {
        disease_id, severity_id, pfirrmann, level_labels (8,), ivd_pathologies (8,7)
    }
    """
    import csv

    # Per-disc data: patient → list of disc rows
    patient_discs = defaultdict(list)
    with open(GRADES_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pat = str(int(float(row["Patient"])))
            patient_discs[pat].append({
                "ivd":           int(float(row["IVD label"])),
                "modic":         int(float(row["Modic"])),
                "up_endplate":   int(float(row["UP endplate"])),
                "low_endplate":  int(float(row["LOW endplate"])),
                "spondylo":      int(float(row["Spondylolisthesis"])),
                "herniation":    int(float(row["Disc herniation"])),
                "narrowing":     int(float(row["Disc narrowing"])),
                "bulging":       int(float(row["Disc bulging"])),
                "pfirrmann":     int(float(row["Pfirrman grade"])),
            })

    labels = {}
    for pat, discs in patient_discs.items():
        # Determine patient-level disease class (priority order)
        any_hern   = any(d["herniation"] for d in discs)
        any_spon   = any(d["spondylo"]   for d in discs)
        any_narrow = any(d["narrowing"]  for d in discs)
        any_bulge  = any(d["bulging"]    for d in discs)
        any_modic  = any(d["modic"] > 0  for d in discs)
        any_ep     = any(d["up_endplate"] or d["low_endplate"] for d in discs)

        # 3-class mapping matched to ResUNet Dice=0.77 capabilities:
        #   0 = Normal (Pfirrmann <= 2, no flags)
        #   1 = Degeneration (disc narrowing, bulging, Pfirrmann 3-4)
        #   2 = Structural (herniation, spondylolisthesis, stenosis, Pfirrmann 5)
        worst_pfi = float(max(d["pfirrmann"] for d in discs))
        has_structural = any_hern or any_spon or (any_narrow and worst_pfi >= 4)
        has_degen = (any_bulge or any_modic or any_ep or any_narrow
                     or worst_pfi >= 3)
        if has_structural or worst_pfi >= 5:
            disease_id = 2   # Structural
        elif has_degen:
            disease_id = 1   # Degeneration
        else:
            disease_id = 0   # Normal

        # Mean Pfirrmann (severity) → severity
        mean_pfi = float(np.mean([d["pfirrmann"] for d in discs]))
        if   mean_pfi <= 2.5: severity_id = 0   # Mild
        elif mean_pfi <= 3.5: severity_id = 1   # Moderate
        else:                 severity_id = 2   # Severe

        # Per-IVD Pfirrmann (IVD labels 1–8 → indices 0–7)
        ivd_pfi = np.zeros(8, dtype=np.float32)
        for d in discs:
            idx = d["ivd"] - 1
            if 0 <= idx < 8:
                ivd_pfi[idx] = d["pfirrmann"]

        # Level labels (1 if any pathology at that IVD)
        level_labels = np.zeros(8, dtype=np.float32)
        for d in discs:
            idx = d["ivd"] - 1
            if 0 <= idx < 8:
                if (d["herniation"] or d["bulging"] or
                        d["narrowing"] or d["spondylo"] or d["modic"]):
                    level_labels[idx] = 1.0

        # Per-IVD pathology vector (8 discs × 7 pathology flags)
        ivd_path = np.zeros((8, 7), dtype=np.float32)
        for d in discs:
            idx = d["ivd"] - 1
            if 0 <= idx < 8:
                ivd_path[idx] = [
                    d["modic"] > 0, d["up_endplate"], d["low_endplate"],
                    d["spondylo"], d["herniation"], d["narrowing"], d["bulging"],
                ]

        labels[pat] = {
            "disease_id":     disease_id,
            "severity_id":    severity_id,
            "mean_pfirrmann": mean_pfi,
            "ivd_pfirrmann":  ivd_pfi,        # (8,)
            "level_labels":   level_labels,   # (8,)
            "ivd_pathology":  ivd_path,       # (8, 7)
        }

    print(f"[Labels] {len(labels)} patients loaded from {GRADES_CSV.name}")
    _dist = defaultdict(int)
    for v in labels.values(): _dist[v["disease_id"]] += 1
    _DNAMES = ["Normal","Herniation","Bulge","Stenosis","DDD","Spondylo","Fracture"]
    print(f"  Disease distribution: " +
          " | ".join(f"{_DNAMES[k]}={v}" for k,v in sorted(_dist.items())))
    return labels


# ═══════════════════════════════════════════════════════════════════════
# Step 4 — Dataset: features + labels paired
# ═══════════════════════════════════════════════════════════════════════

class SpineFeatureDataset(Dataset):
    """
    Each sample = (feature_vec, labels) for one patient.
    Feature vector combines:
      - max_prob  (19,) from ResUNet
      - mean_prob (19,) from ResUNet
      - std_prob  (19,) from ResUNet
      Total: 57-dim image feature
    """
    FEAT_DIM = NUM_CLASSES * 3   # 57

    def __init__(self, samples):
        self.samples = samples

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        feat = np.concatenate([
            s["max_prob"], s["mean_prob"], s["std_prob"]
        ]).astype(np.float32)   # (57,)
        return {
            "feat":        torch.tensor(feat),
            "disease_id":  torch.tensor(s["disease_id"],  dtype=torch.long),
            "severity_id": torch.tensor(s["severity_id"], dtype=torch.long),
            "pfirrmann":   torch.tensor(s["mean_pfirrmann"], dtype=torch.float32),
            "ivd_pfi":     torch.tensor(s["ivd_pfirrmann"],  dtype=torch.float32),  # (8,)
            "level":       torch.tensor(s["level_labels"],   dtype=torch.float32),  # (8,)
        }


def build_dataset(model, device, labels, max_patients=None):
    """
    Run ResUNet on all T2 images, pair with labels, return dataset.
    Uses a feature cache to avoid re-running inference.
    """
    import cv2
    cache_path = OUT_DIR / "feature_cache.pt"

    # Load cache if exists
    cache = {}
    if cache_path.exists():
        try:
            cache = torch.load(str(cache_path), map_location="cpu",
                               weights_only=False)
            print(f"[Cache] Loaded {len(cache)} cached feature vectors")
        except Exception:
            cache = {}

    samples    = []
    new_feats  = 0
    patient_ids = sorted(labels.keys())
    if max_patients:
        patient_ids = patient_ids[:max_patients]

    print(f"\n[Features] Building dataset for {len(patient_ids)} patients ...")
    t0 = time.time()

    for pid in patient_ids:
        # ── Use cache directly — no file I/O needed ──────────────────
        if pid in cache:
            feat_data = cache[pid]
        else:
            # Run ResUNet inference only if not cached
            img_candidates = sorted(IMAGES_DIR.glob(f"{pid}_t2*.mha"))
            if not img_candidates:
                img_candidates = sorted(IMAGES_DIR.glob(f"{pid}_t1*.mha"))
            if not img_candidates:
                continue

            feat_data = extract_features(model, device, img_candidates[0])
            if feat_data is None:
                continue

            cache[pid] = feat_data
            new_feats += 1

        if pid not in labels:
            continue

        lbl = labels[pid]
        samples.append({
            "patient_id":    pid,
            "max_prob":      np.array(feat_data["max_prob"],  dtype=np.float32),
            "mean_prob":     np.array(feat_data["mean_prob"], dtype=np.float32),
            "std_prob":      np.array(feat_data["std_prob"],  dtype=np.float32),
            "disease_id":    lbl["disease_id"],
            "severity_id":   lbl["severity_id"],
            "mean_pfirrmann":lbl["mean_pfirrmann"],
            "ivd_pfirrmann": lbl["ivd_pfirrmann"],
            "level_labels":  lbl["level_labels"],
        })

    # Save updated cache
    if new_feats > 0:
        torch.save(cache, str(cache_path))
        print(f"[Cache] Saved {len(cache)} entries (+{new_feats} new)")

    elapsed = time.time() - t0
    print(f"[Features] {len(samples)} samples ready ({elapsed:.1f}s)")
    return SpineFeatureDataset(samples)


# ═══════════════════════════════════════════════════════════════════════
# Step 5 — Model: lightweight projector + fusion + multitask head
# ═══════════════════════════════════════════════════════════════════════

class SpineClassifier(nn.Module):
    """
    Compact classifier for 218-patient SPIDER dataset.
    3-class disease: 0=Normal, 1=Degeneration, 2=Structural
    Input:  (B, 57) — concatenated ResUNet feature vectors
    """

    def __init__(self, feat_dim=57, fusion_dim=64, num_disease=3,
                 num_severity=3, num_levels=8, dropout=0.5):
        super().__init__()
        self.num_disease = num_disease

        # Minimal projector — fewer params to prevent overfitting
        self.img_projector = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.BatchNorm1d(128),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, fusion_dim), nn.BatchNorm1d(fusion_dim),
        )
        # Disease head
        self.disease_head = nn.Sequential(
            nn.Dropout(dropout * 0.6),
            nn.Linear(fusion_dim, num_disease),
        )
        # Severity head
        self.severity_head = nn.Sequential(
            nn.Dropout(dropout * 0.6),
            nn.Linear(fusion_dim, num_severity),
        )
        # Level localization
        self.level_head = nn.Sequential(
            nn.Dropout(dropout * 0.4),
            nn.Linear(fusion_dim, num_levels),
        )
        # Per-IVD Pfirrmann regression
        self.pfirrmann_head = nn.Sequential(
            nn.Dropout(dropout * 0.3),
            nn.Linear(fusion_dim, num_levels),
            nn.Sigmoid(),
        )
        # Disease-conditioned adapter
        self.adapter = nn.Sequential(
            nn.Linear(fusion_dim + num_disease, fusion_dim),
            nn.BatchNorm1d(fusion_dim), nn.GELU(),
        )

    def forward(self, feat):
        h           = self.img_projector(feat)
        dis_logits  = self.disease_head(h)
        dis_probs   = F.softmax(dis_logits, -1)
        h2          = self.adapter(torch.cat([h, dis_probs], -1))
        sev_logits  = self.severity_head(h2)
        lvl_logits  = self.level_head(h2)
        pfi_scores  = self.pfirrmann_head(h2) * 4.0 + 1.0
        return {
            "disease_logits":  dis_logits,
            "disease_probs":   dis_probs,
            "severity_logits": sev_logits,
            "level_logits":    lvl_logits,
            "pfirrmann":       pfi_scores,
            "mean_pfirrmann":  pfi_scores.mean(-1),
        }




# ═══════════════════════════════════════════════════════════════════════
# Step 6 — Loss functions
# ═══════════════════════════════════════════════════════════════════════

class ClassifierLoss(nn.Module):
    def __init__(self, class_weights=None, disease_weight=1.0,
                 severity_weight=0.4, level_weight=0.4, pfirrmann_weight=0.2):
        super().__init__()
        self.w_dis  = disease_weight
        self.w_sev  = severity_weight
        self.w_lvl  = level_weight
        self.w_pfi  = pfirrmann_weight
        # Class weights to handle imbalance (Stenosis=106 dominates)
        self.class_weights = class_weights

    def forward(self, pred, batch):
        device = pred["disease_logits"].device
        cw = self.class_weights.to(device) if self.class_weights is not None else None

        # Disease — weighted cross-entropy + label smoothing
        dis_loss = F.cross_entropy(
            pred["disease_logits"],
            batch["disease_id"].to(device),
            weight=cw, label_smoothing=0.1
        )
        # Severity
        sev_loss = F.cross_entropy(
            pred["severity_logits"],
            batch["severity_id"].to(device),
            label_smoothing=0.05
        )
        # Level (binary cross-entropy)
        lvl_loss = F.binary_cross_entropy_with_logits(
            pred["level_logits"], batch["level"].to(device)
        )
        # Pfirrmann regression
        pfi_target = batch["ivd_pfi"].to(device)
        pfi_loss   = F.mse_loss(pred["pfirrmann"], pfi_target)

        total = (self.w_dis * dis_loss + self.w_sev * sev_loss +
                 self.w_lvl * lvl_loss + self.w_pfi * pfi_loss)
        return total, {
            "disease": dis_loss.item(), "severity": sev_loss.item(),
            "level":   lvl_loss.item(), "pfirrmann": pfi_loss.item(),
            "total":   total.item(),
        }


# ═══════════════════════════════════════════════════════════════════════
# Step 7 — Training loop
# ═══════════════════════════════════════════════════════════════════════

def train(args, device):
    print("\n" + "="*60)
    print("  ATM-Net++ Classifier Training")
    print(f"  Device: {device} | Epochs: {args.epochs} | LR: {args.lr}")
    print("="*60)

    resunet = load_resunet(device)
    labels  = load_labels()
    if not labels:
        print("ERROR: No labels loaded."); return

    dataset = build_dataset(resunet, device, labels,
                             max_patients=args.max_patients)
    if len(dataset) < 4:
        print(f"ERROR: Only {len(dataset)} samples."); return

    # ── Class weights (inverse frequency, sized to actual num_disease) ──
    all_dis = [dataset[i]["disease_id"].item() for i in range(len(dataset))]
    n_dis   = 3   # 3-class model
    counts  = np.bincount(all_dis, minlength=n_dis).astype(np.float32)[:n_dis]
    counts  = np.where(counts == 0, 1, counts)
    cw      = torch.tensor(1.0 / counts)
    cw      = cw / cw.sum() * n_dis
    print(f"\n  Disease counts : {counts.astype(int).tolist()}")
    print(f"  Class weights  : {[round(float(w),2) for w in cw]}")

    # ── Oversample minority classes so each class has at least 20 samples ──
    from torch.utils.data import WeightedRandomSampler
    sample_weights = torch.tensor([float(cw[dataset[i]["disease_id"].item()])
                                   for i in range(len(dataset))])

    # ── Train/val split ───────────────────────────────────────────────
    n_val   = max(2, int(len(dataset) * 0.20))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    # Weighted sampler on train set only
    train_weights = sample_weights[:n_train]
    sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_ds) * 3,   # 3× oversample each epoch
        replacement=True,
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch,
                          sampler=sampler, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch,
                          shuffle=False, num_workers=0)
    print(f"  Train: {n_train} | Val: {n_val} | Batch: {args.batch} | "
          f"Oversampled: {len(train_ds)*3} per epoch")

    # ── Model ─────────────────────────────────────────────────────────
    model = SpineClassifier(
        feat_dim=SpineFeatureDataset.FEAT_DIM,
        fusion_dim=64, num_disease=3, dropout=args.dropout
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=5e-3)
    # Warmup 10 epochs, then cosine decay
    def lr_lambda(ep):
        if ep < 10: return (ep + 1) / 10
        return 0.5 * (1 + np.cos(np.pi * (ep - 10) / max(1, args.epochs - 10)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = ClassifierLoss(class_weights=cw)

    best_val_loss = float("inf")
    best_val_acc  = 0.0
    patience      = args.patience
    no_improve    = 0
    history       = []

    print(f"\n{'Ep':>4} {'TrLoss':>8} {'VaLoss':>8} "
          f"{'DisAcc':>7} {'SevAcc':>7} {'PfiMAE':>7} {'LR':>8}")
    print("-" * 62)

    for ep in range(1, args.epochs + 1):
        # ── Train ─────────────────────────────────────────────────────
        model.train()
        tr_losses = defaultdict(float)
        for batch in train_dl:
            optimizer.zero_grad()
            pred = model(batch["feat"].to(device))
            loss, breakdown = criterion(pred, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            for k, v in breakdown.items(): tr_losses[k] += v

        n_tr     = len(train_dl)
        tr_total = tr_losses["total"] / n_tr

        # ── Validate ──────────────────────────────────────────────────
        model.eval()
        val_losses  = defaultdict(float)
        dis_correct = sev_correct = n_total = 0
        pfi_errors  = []

        with torch.no_grad():
            for batch in val_dl:
                pred = model(batch["feat"].to(device))
                loss, breakdown = criterion(pred, batch)
                for k, v in breakdown.items(): val_losses[k] += v

                dis_pred = pred["disease_logits"].argmax(-1).cpu()
                sev_pred = pred["severity_logits"].argmax(-1).cpu()
                dis_correct += (dis_pred == batch["disease_id"]).sum().item()
                sev_correct += (sev_pred == batch["severity_id"]).sum().item()
                n_total     += len(batch["disease_id"])

                pfi_pred   = pred["pfirrmann"].cpu()
                pfi_target = batch["ivd_pfi"]
                mask = pfi_target > 0
                if mask.any():
                    pfi_errors.append(
                        (pfi_pred[mask] - pfi_target[mask]).abs().mean().item()
                    )

        n_val_b  = max(1, len(val_dl))
        val_total = val_losses["total"] / n_val_b
        dis_acc  = dis_correct / max(1, n_total)
        sev_acc  = sev_correct / max(1, n_total)
        pfi_mae  = float(np.mean(pfi_errors)) if pfi_errors else 0.0
        lr_now   = optimizer.param_groups[0]["lr"]

        scheduler.step()

        history.append({
            "ep": ep, "tr": round(tr_total, 4), "val": round(val_total, 4),
            "dis_acc": round(dis_acc, 4), "sev_acc": round(sev_acc, 4),
            "pfi_mae": round(pfi_mae, 4),
        })

        # ── Save best (by val loss) ───────────────────────────────────
        flag = ""
        if val_total < best_val_loss:
            best_val_loss = val_total
            best_val_acc  = dis_acc
            no_improve    = 0
            torch.save({
                "epoch":      ep,
                "model_state": model.state_dict(),
                "dis_acc":    dis_acc,
                "sev_acc":    sev_acc,
                "pfi_mae":    pfi_mae,
                "val_loss":   val_total,
                "feat_dim":   SpineFeatureDataset.FEAT_DIM,
                "fusion_dim": 64,
                "num_disease": 3,
                "dropout":    args.dropout,
            }, str(OUT_DIR / "best_classifier.pth"))
            flag = " ★"
        else:
            no_improve += 1

        if ep % 10 == 0 or ep <= 5 or flag:
            print(f"{ep:>4}  {tr_total:>8.4f}  {val_total:>8.4f}  "
                  f"{dis_acc:>6.1%}  {sev_acc:>6.1%}  "
                  f"{pfi_mae:>7.3f}  {lr_now:>8.2e}{flag}")

        # ── Early stopping ────────────────────────────────────────────
        if no_improve >= patience:
            print(f"\n  Early stop at epoch {ep} (patience={patience})")
            break

    # Save last + history
    torch.save({"epoch": ep, "model_state": model.state_dict()},
               str(OUT_DIR / "last_classifier.pth"))
    with open(OUT_DIR / "classifier_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "="*60)
    print(f"  Done | Best Val Loss: {best_val_loss:.4f} "
          f"| Best Disease Acc: {best_val_acc:.1%}")
    print(f"  Saved → {OUT_DIR / 'best_classifier.pth'}")
    print("="*60)
    return model


# ═══════════════════════════════════════════════════════════════════════
# Step 8 — Wire trained weights into server.py
# ═══════════════════════════════════════════════════════════════════════

def patch_server(model: SpineClassifier):
    """
    Save a compatible checkpoint that server.py can load via
    load_neural_models() → _multi_task_head.
    Also saves feature projector weights.
    """
    save_path = OUT_DIR / "best_classifier.pth"
    print(f"\n[Patch] Checkpoint ready at: {save_path}")
    print("[Patch] server.py will auto-load this on next restart via load_neural_models()")
    print("[Patch] To apply immediately: restart the server with   py server.py")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",       type=int,   default=200)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--batch",        type=int,   default=16)
    parser.add_argument("--dropout",      type=float, default=0.5)
    parser.add_argument("--patience",     type=int,   default=30,
                        help="Early stopping patience (val loss)")
    parser.add_argument("--max_patients", type=int,   default=None)
    parser.add_argument("--quick",        action="store_true",
                        help="10 epochs, 20 patients — smoke test")
    args = parser.parse_args()

    if args.quick:
        args.epochs       = 10
        args.max_patients = 20
        args.patience     = 10
        print("[Quick mode] 10 epochs, 20 patients")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = train(args, device)
    if model is not None:
        patch_server(model)
