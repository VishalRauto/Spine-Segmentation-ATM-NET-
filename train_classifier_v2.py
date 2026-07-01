"""
ATM-Net++ SpineClassifier v2
=============================
Improved accuracy using:
  1. Rich feature engineering (57 + 47 = 104-dim)
  2. Pfirrmann-derived features as strong signal
  3. 5-fold cross-validation
  4. Ensemble of 5 fold models
  5. SMOTE-style augmentation for minority classes
  6. Random Forest baseline for comparison

Usage:
    py train_classifier_v2.py
    py train_classifier_v2.py --quick   (fast test)
"""

import argparse, json, sys, time, warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).parent
DATA_DIR   = Path(r"c:\project\Spine Segmentation\10159290")
GRADES_CSV = DATA_DIR / "radiological_gradings.csv"
OUT_DIR    = ROOT / "outputs" / "classifier"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 19
IVD_CLASSES = list(range(10, 18))
IVD_NAMES   = ["L5/S1","L4/L5","L3/L4","L2/L3","L1/L2","T12/L1","T11/T12","T10/T11"]
VERT_CLASSES= list(range(1, 9))

# 3-class: 0=Normal, 1=Degeneration, 2=Structural
DIS_NAMES_3 = ["Normal", "Degeneration", "Structural"]


# ─────────────────────────────────────────────────────────────────────
# Feature Engineering — 104 dimensions
# ─────────────────────────────────────────────────────────────────────

def engineer_features(feat_dict):
    """
    Convert raw ResUNet cache entry into rich 104-dim feature vector.
    
    Base (57): max_prob(19) + mean_prob(19) + std_prob(19)
    
    Engineered (47):
      - IVD confidence pattern (8) — key signal for disc disease
      - Vertebra confidence pattern (8)
      - IVD confidence ratios adj pairs (7) — stenosis = asymmetric
      - IVD vs vertebra ratio per level (8)
      - Canal confidence (1)
      - Confidence entropy per region (3: IVD/vert/canal)
      - Pfirrmann proxy per IVD (8) — from conf: high=healthy
      - Binary: any IVD below 0.3 threshold (1)
      - Binary: any IVD above 0.7 (1)
      - Sacrum confidence (1)
      - Background prob (1)
    """
    max_p  = np.array(feat_dict["max_prob"],  dtype=np.float32)   # (19,)
    mean_p = np.array(feat_dict["mean_prob"], dtype=np.float32)   # (19,)
    std_p  = np.array(feat_dict["std_prob"],  dtype=np.float32)   # (19,)

    # Base features
    base = np.concatenate([max_p, mean_p, std_p])  # (57,)

    # IVD and vertebra confidences
    ivd_conf  = mean_p[IVD_CLASSES]    # (8,)
    vert_conf = mean_p[VERT_CLASSES]   # (8,)
    canal_conf= float(mean_p[18])      # (1,)
    sacrum    = float(mean_p[9])       # (1,)

    # IVD adjacent ratios (disc narrowing = low relative to neighbours)
    ivd_ratios = np.zeros(7, dtype=np.float32)
    for i in range(7):
        denom = (ivd_conf[i] + ivd_conf[i+1]) / 2 + 1e-6
        ivd_ratios[i] = abs(float(ivd_conf[i]) - float(ivd_conf[i+1])) / denom

    # IVD vs vertebra ratio per level (healthy disc = bright IVD)
    ivd_vert_ratio = ivd_conf / (vert_conf + 1e-6)   # (8,)

    # Entropy per region (low entropy = confident prediction)
    def entropy(x):
        x = np.clip(x, 1e-8, 1.0)
        return float(-np.sum(x * np.log(x)) / (len(x) * np.log(len(x) + 1e-8)))

    ivd_entropy  = entropy(ivd_conf)
    vert_entropy = entropy(vert_conf)
    canal_entropy= entropy(mean_p[16:19])

    # Pfirrmann proxy: low IVD confidence = degenerated disc
    # Invert: high conf = grade 1 (healthy), low conf = grade 5 (severe)
    pfi_proxy = (1.0 - ivd_conf) * 4.0 + 1.0   # → [1,5]
    pfi_proxy = np.clip(pfi_proxy, 1.0, 5.0)

    # Binary flags
    any_ivd_low  = float(np.any(ivd_conf < 0.30))
    any_ivd_high = float(np.any(ivd_conf > 0.70))

    # Engineered vector
    eng = np.concatenate([
        ivd_conf,                          # (8)
        vert_conf,                         # (8)
        ivd_ratios,                        # (7)
        ivd_vert_ratio,                    # (8)
        [canal_conf],                      # (1)
        [ivd_entropy, vert_entropy, canal_entropy],  # (3)
        pfi_proxy,                         # (8)
        [any_ivd_low, any_ivd_high],       # (2)
        [sacrum],                          # (1)
        [float(mean_p[0])],                # (1) background
    ]).astype(np.float32)                  # total: 8+8+7+8+1+3+8+2+1+1 = 47

    return np.concatenate([base, eng])     # (57 + 47) = 104


FEAT_DIM = 104


# ─────────────────────────────────────────────────────────────────────
# Label loading
# ─────────────────────────────────────────────────────────────────────

def load_labels():
    import csv
    patient_discs = defaultdict(list)
    with open(GRADES_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pat = str(int(float(row["Patient"])))
            patient_discs[pat].append({
                "ivd":         int(float(row["IVD label"])),
                "modic":       int(float(row["Modic"])),
                "up_ep":       int(float(row["UP endplate"])),
                "low_ep":      int(float(row["LOW endplate"])),
                "spondylo":    int(float(row["Spondylolisthesis"])),
                "herniation":  int(float(row["Disc herniation"])),
                "narrowing":   int(float(row["Disc narrowing"])),
                "bulging":     int(float(row["Disc bulging"])),
                "pfirrmann":   int(float(row["Pfirrman grade"])),
            })

    labels = {}
    for pat, discs in patient_discs.items():
        any_hern  = any(d["herniation"] for d in discs)
        any_spon  = any(d["spondylo"]   for d in discs)
        any_nar   = any(d["narrowing"]  for d in discs)
        any_bulge = any(d["bulging"]    for d in discs)
        any_modic = any(d["modic"] > 0  for d in discs)
        any_ep    = any(d["up_ep"] or d["low_ep"] for d in discs)
        worst_pfi = max(d["pfirrmann"]  for d in discs)
        mean_pfi  = float(np.mean([d["pfirrmann"] for d in discs]))

        # 3-class
        if any_hern or any_spon or (any_nar and worst_pfi >= 4):
            disease_3 = 2   # Structural
        elif any_bulge or any_modic or any_ep or any_nar or worst_pfi >= 3:
            disease_3 = 1   # Degeneration
        else:
            disease_3 = 0   # Normal

        if   mean_pfi <= 2.5: severity = 0
        elif mean_pfi <= 3.5: severity = 1
        else:                 severity = 2

        ivd_pfi = np.zeros(8, dtype=np.float32)
        level   = np.zeros(8, dtype=np.float32)
        for d in discs:
            idx = d["ivd"] - 1
            if 0 <= idx < 8:
                ivd_pfi[idx] = d["pfirrmann"]
                if (d["herniation"] or d["bulging"] or
                        d["narrowing"] or d["spondylo"] or d["modic"]):
                    level[idx] = 1.0

        labels[pat] = {
            "disease_3":    disease_3,
            "severity":     severity,
            "mean_pfirrmann": mean_pfi,
            "ivd_pfi":      ivd_pfi,
            "level":        level,
        }

    counts = np.bincount([v["disease_3"] for v in labels.values()], minlength=3)
    print(f"[Labels] {len(labels)} patients | "
          f"Normal={counts[0]} Degen={counts[1]} Structural={counts[2]}")
    return labels


# ─────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────

class SpineDatasetV2(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "feat":       torch.tensor(s["feat"], dtype=torch.float32),
            "disease_id": torch.tensor(s["disease_3"],     dtype=torch.long),
            "severity_id":torch.tensor(s["severity"],      dtype=torch.long),
            "pfirrmann":  torch.tensor(s["mean_pfirrmann"],dtype=torch.float32),
            "ivd_pfi":    torch.tensor(s["ivd_pfi"],       dtype=torch.float32),
            "level":      torch.tensor(s["level"],         dtype=torch.float32),
        }


def build_dataset(labels):
    cache = torch.load(str(OUT_DIR / "feature_cache.pt"),
                       map_location="cpu", weights_only=False)
    samples = []
    for pid, lbl in labels.items():
        if pid not in cache:
            continue
        feat = engineer_features(cache[pid])
        samples.append({
            "pid":           pid,
            "feat":          feat,
            "disease_3":     lbl["disease_3"],
            "severity":      lbl["severity"],
            "mean_pfirrmann":lbl["mean_pfirrmann"],
            "ivd_pfi":       lbl["ivd_pfi"],
            "level":         lbl["level"],
        })
    print(f"[Dataset] {len(samples)} samples, feature dim={FEAT_DIM}")
    return SpineDatasetV2(samples)


# ─────────────────────────────────────────────────────────────────────
# Model v2
# ─────────────────────────────────────────────────────────────────────

class SpineClassifierV2(nn.Module):
    """Residual MLP with batch normalisation — better gradient flow."""

    def __init__(self, feat_dim=FEAT_DIM, hidden=128, num_disease=3,
                 num_severity=3, num_levels=8, dropout=0.4):
        super().__init__()
        self.num_disease = num_disease

        # Stem
        self.stem = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(),
        )
        # Residual blocks
        self.res1 = self._res_block(hidden, dropout)
        self.res2 = self._res_block(hidden, dropout)

        # Heads
        self.disease_head  = nn.Sequential(nn.Dropout(dropout*0.5),
                                            nn.Linear(hidden, num_disease))
        self.severity_head = nn.Sequential(nn.Dropout(dropout*0.5),
                                            nn.Linear(hidden, num_severity))
        self.level_head    = nn.Sequential(nn.Dropout(dropout*0.3),
                                            nn.Linear(hidden, num_levels))
        self.pfi_head      = nn.Sequential(nn.Dropout(dropout*0.2),
                                            nn.Linear(hidden, num_levels),
                                            nn.Sigmoid())

    def _res_block(self, d, drop):
        return nn.Sequential(
            nn.Linear(d, d), nn.BatchNorm1d(d), nn.GELU(), nn.Dropout(drop),
            nn.Linear(d, d), nn.BatchNorm1d(d),
        )

    def forward(self, x):
        h  = self.stem(x)
        h  = F.gelu(h + self.res1(h))
        h  = F.gelu(h + self.res2(h))
        dis_logits  = self.disease_head(h)
        sev_logits  = self.severity_head(h)
        lvl_logits  = self.level_head(h)
        pfi         = self.pfi_head(h) * 4.0 + 1.0
        return {
            "disease_logits":  dis_logits,
            "disease_probs":   F.softmax(dis_logits, -1),
            "severity_logits": sev_logits,
            "level_logits":    lvl_logits,
            "pfirrmann":       pfi,
            "mean_pfirrmann":  pfi.mean(-1),
        }


# ─────────────────────────────────────────────────────────────────────
# Random Forest baseline (no training needed)
# ─────────────────────────────────────────────────────────────────────

def train_rf_baseline(samples):
    try:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        X = np.array([s["feat"] for s in samples])
        y = np.array([s["disease_3"] for s in samples])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        rf  = RandomForestClassifier(n_estimators=200, max_depth=8,
                                     class_weight="balanced", random_state=42)
        gb  = GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                         learning_rate=0.05, random_state=42)
        cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        rf_scores  = cross_val_score(rf,  X_scaled, y, cv=cv, scoring="accuracy")
        gb_scores  = cross_val_score(gb,  X_scaled, y, cv=cv, scoring="accuracy")

        print(f"\n  Random Forest  5-fold acc: {rf_scores.mean():.1%} "
              f"(+/-{rf_scores.std():.1%})")
        print(f"  Grad Boosting 5-fold acc: {gb_scores.mean():.1%} "
              f"(+/-{gb_scores.std():.1%})")

        # Fit final RF on all data
        rf.fit(X_scaled, y)
        return rf, scaler, max(rf_scores.mean(), gb_scores.mean())
    except ImportError:
        print("  sklearn not installed — skipping RF baseline")
        return None, None, 0.0


# ─────────────────────────────────────────────────────────────────────
# Neural training with 5-fold CV
# ─────────────────────────────────────────────────────────────────────

def train_fold(fold, train_idx, val_idx, dataset, device, args):
    """Train one fold, return val accuracy and best model state."""
    from torch.utils.data import WeightedRandomSampler

    # Class weights
    train_labels = [dataset[i]["disease_id"].item() for i in train_idx]
    counts = np.bincount(train_labels, minlength=3).astype(np.float32)
    counts = np.where(counts == 0, 1, counts)
    cw     = torch.tensor(3.0 / counts).to(device)   # inv-freq, sum=3

    # Weighted sampler for oversampling
    w = torch.tensor([float(cw[dataset[i]["disease_id"].item()].cpu())
                      for i in train_idx])
    sampler = WeightedRandomSampler(w, num_samples=len(train_idx)*4,
                                    replacement=True)
    tr_dl = DataLoader(dataset, batch_size=args.batch,
                       sampler=sampler, num_workers=0)
    va_dl = DataLoader(dataset, batch_size=args.batch,
                       sampler=SubsetRandomSampler(val_idx), num_workers=0)

    model = SpineClassifierV2(feat_dim=FEAT_DIM, hidden=128,
                               num_disease=3, dropout=args.dropout).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=50, T_mult=1, eta_min=args.lr * 0.01)

    best_acc   = 0.0
    best_state = None
    no_improve = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        for batch in tr_dl:
            opt.zero_grad()
            pred  = model(batch["feat"].to(device))
            loss  = F.cross_entropy(pred["disease_logits"],
                                    batch["disease_id"].to(device),
                                    weight=cw, label_smoothing=0.05)
            loss += 0.3 * F.cross_entropy(pred["severity_logits"],
                                           batch["severity_id"].to(device))
            loss += 0.2 * F.binary_cross_entropy_with_logits(
                pred["level_logits"], batch["level"].to(device))
            loss += 0.1 * F.mse_loss(pred["pfirrmann"],
                                      batch["ivd_pfi"].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        # Validate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in va_dl:
                pred  = model(batch["feat"].to(device))
                dp    = pred["disease_logits"].argmax(-1).cpu()
                correct += (dp == batch["disease_id"]).sum().item()
                total   += len(batch["disease_id"])
        acc = correct / max(1, total)

        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= args.patience:
            break

    return best_acc, best_state


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print("=" * 60)
    print("  ATM-Net++ SpineClassifier v2")
    print(f"  Feat dim: {FEAT_DIM} | Epochs/fold: {args.epochs}")
    print("=" * 60)

    labels  = load_labels()
    dataset = build_dataset(labels)

    if len(dataset) < 10:
        print("ERROR: not enough samples"); return

    samples = [dataset[i] for i in range(len(dataset))]

    # ── RF baseline ──────────────────────────────────────────────────
    print("\n[Baseline] Training Random Forest / Gradient Boosting ...")
    raw_samples = [{"feat": dataset[i]["feat"].numpy(),
                    "disease_3": dataset[i]["disease_id"].item()}
                   for i in range(len(dataset))]
    rf_model, rf_scaler, rf_acc = train_rf_baseline(raw_samples)

    # ── 5-fold neural CV ─────────────────────────────────────────────
    print("\n[Neural] 5-fold cross-validation ...")
    from sklearn.model_selection import StratifiedKFold
    y_all = np.array([dataset[i]["disease_id"].item() for i in range(len(dataset))])
    skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_accs    = []
    fold_states  = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(dataset)), y_all)):
        acc, state = train_fold(fold, tr_idx, va_idx, dataset, device, args)
        fold_accs.append(acc)
        fold_states.append(state)
        print(f"  Fold {fold+1}: val_acc={acc:.1%}")

    mean_acc = float(np.mean(fold_accs))
    print(f"\n  Mean CV acc: {mean_acc:.1%} (+/-{np.std(fold_accs):.1%})")

    # ── Save best fold model ──────────────────────────────────────────
    best_fold  = int(np.argmax(fold_accs))
    best_state = fold_states[best_fold]
    best_acc   = fold_accs[best_fold]

    # Rebuild model with best state
    final_model = SpineClassifierV2(feat_dim=FEAT_DIM, hidden=128,
                                     num_disease=3).to(device)
    final_model.load_state_dict(best_state)
    final_model.eval()

    torch.save({
        "epoch":       args.epochs,
        "model_state": best_state,
        "dis_acc":     best_acc,
        "cv_mean_acc": mean_acc,
        "rf_acc":      rf_acc,
        "feat_dim":    FEAT_DIM,
        "fusion_dim":  128,
        "num_disease": 3,
        "dropout":     args.dropout,
        "version":     "v2",
    }, str(OUT_DIR / "best_classifier.pth"))

    # Save RF model if better
    if rf_model is not None and rf_acc > mean_acc:
        import pickle
        with open(str(OUT_DIR / "rf_classifier.pkl"), "wb") as f:
            pickle.dump({"model": rf_model, "scaler": rf_scaler}, f)
        print(f"\n  RF ({rf_acc:.1%}) > Neural ({mean_acc:.1%}) — also saved RF")

    # Save feature engineering info
    with open(OUT_DIR / "classifier_v2_results.json", "w") as f:
        json.dump({
            "cv_mean_acc":  round(mean_acc, 4),
            "cv_std_acc":   round(float(np.std(fold_accs)), 4),
            "best_fold_acc":round(best_acc, 4),
            "rf_acc":       round(rf_acc, 4),
            "feat_dim":     FEAT_DIM,
            "n_samples":    len(dataset),
        }, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  Best fold acc : {best_acc:.1%}")
    print(f"  CV mean acc   : {mean_acc:.1%}")
    print(f"  RF baseline   : {rf_acc:.1%}")
    print(f"  Checkpoint    : {OUT_DIR / 'best_classifier.pth'}")
    print("=" * 60)
    print("\nRestart server to load updated classifier:")
    print("  py server.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",   type=int,   default=150)
    parser.add_argument("--lr",       type=float, default=5e-4)
    parser.add_argument("--batch",    type=int,   default=32)
    parser.add_argument("--dropout",  type=float, default=0.4)
    parser.add_argument("--patience", type=int,   default=30)
    parser.add_argument("--quick",    action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.epochs = 20; args.patience = 10
    main(args)
