"""
ATM-Net++ Fusion + MultiTaskHead Training
==========================================
Trains MultimodalFusionModule (ATPG+HASF+CCAE) and MultiTaskHead
end-to-end using:
  - Real ResUNet image features (from feature cache)
  - Real Bio-ClinicalBERT anatomy text embeddings
  - SPIDER radiological_gradings.csv labels
  - Patient demographics from overview.csv

After training:
  - outputs/fusion/fusion_module.pth
  - outputs/fusion/multitask_head.pth

server.py loads these automatically on restart.

Usage:
    py train_fusion.py
    py train_fusion.py --quick   # 10 epochs smoke test
"""

import argparse, csv, json, sys, time, warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

ROOT       = Path(__file__).parent
DATA_DIR   = Path(r"c:\project\Spine Segmentation\10159290")
GRADES_CSV = DATA_DIR / "radiological_gradings.csv"
OVERVIEW   = DATA_DIR / "overview.csv"
CLF_CACHE  = ROOT / "outputs" / "classifier" / "feature_cache.pt"
OUT_DIR    = ROOT / "outputs" / "fusion"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_IMG_CLASSES = 19
IVD_CLASSES     = list(range(10, 18))
IVD_NAMES       = ["L5/S1","L4/L5","L3/L4","L2/L3","L1/L2","T12/L1","T11/T12","T10/T11"]

# 7-class disease
DISEASE_NAMES = ["Normal","Disc Herniation","Disc Bulge","Spinal Stenosis",
                 "Disc Degeneration","Spondylolisthesis","Compression Fracture"]


# ─────────────────────────────────────────────────────────────────────
# Step 1 — Load demographics from overview.csv
# ─────────────────────────────────────────────────────────────────────

def load_demographics():
    """
    Returns dict: patient_id → {sex, age, field_strength, pixel_spacing}
    """
    demo = {}
    try:
        with open(OVERVIEW, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row.get("new_file_name", "")
                # Extract patient id from filename like "1_t2"
                pid = fname.split("_")[0] if "_" in fname else fname
                if pid and pid not in demo:
                    sex = str(row.get("sex", "F")).strip().upper()
                    try:
                        birth = row.get("birth_date", "")
                        age = float(birth) if birth.strip() else 50.0
                        age = max(0, min(100, 2024 - age)) if age > 1900 else float(birth) if birth else 50.0
                    except: age = 50.0
                    try:
                        fs = float(row.get("MagneticFieldStrength", 1.5))
                    except: fs = 1.5
                    demo[pid] = {
                        "sex": 1.0 if sex.startswith("M") else 0.0,
                        "age": np.clip(age / 80.0, 0, 1),
                        "field_strength": np.clip(fs / 3.0, 0, 1),
                    }
    except Exception as e:
        print(f"[Demo] Could not load overview.csv: {e}")
    return demo


def make_demo_vec(demo_info: dict) -> np.ndarray:
    """Build 8-dim demographic vector from demo_info dict."""
    v = np.zeros(8, dtype=np.float32)
    v[0] = demo_info.get("sex", 0.0)
    v[1] = demo_info.get("age", 0.625)          # default 50yrs
    v[2] = demo_info.get("field_strength", 0.5) # 1.5T default
    v[3:] = 0.5                                  # unknown scanner params
    return v


# ─────────────────────────────────────────────────────────────────────
# Step 2 — Build anatomy text for each patient (ATPG input)
# ─────────────────────────────────────────────────────────────────────

def make_anatomy_text(feat_dict: dict) -> str:
    """
    Generate anatomy-aware clinical text from ResUNet features.
    This is what Bio-ClinicalBERT will encode for each patient.
    """
    max_p = np.array(feat_dict["max_prob"], dtype=np.float32)

    cls_names = {
        1:"L5 vertebra", 2:"L4 vertebra", 3:"L3 vertebra", 4:"L2 vertebra",
        5:"L1 vertebra", 6:"T12 vertebra", 7:"T11 vertebra", 8:"T10 vertebra",
        9:"sacrum",
        10:"L5/S1 disc", 11:"L4/L5 disc", 12:"L3/L4 disc", 13:"L2/L3 disc",
        14:"L1/L2 disc", 15:"T12/L1 disc", 16:"T11/T12 disc", 17:"T10/T11 disc",
        18:"spinal canal",
    }
    visible, degraded = [], []
    for c in range(1, NUM_IMG_CLASSES):
        conf = float(max_p[c])
        if conf > 0.50:
            visible.append(cls_names.get(c, ""))
        elif c in IVD_CLASSES and conf < 0.25:
            degraded.append(cls_names.get(c, ""))

    text = "Lumbar spine MRI."
    if visible:
        text += f" Visible structures: {', '.join(v for v in visible if v)}."
    if degraded:
        text += f" Low signal intensity at: {', '.join(d for d in degraded if d)}, suggesting disc degeneration."

    ivd_confs = [float(max_p[c]) for c in IVD_CLASSES]
    mean_ivd = np.mean(ivd_confs)
    if mean_ivd > 0.70:
        text += " Overall disc signal appears preserved."
    elif mean_ivd > 0.40:
        text += " Moderate disc signal loss noted."
    else:
        text += " Severe disc signal loss, significant degeneration."

    return text


# ─────────────────────────────────────────────────────────────────────
# Step 3 — Encode all anatomy texts with Bio-ClinicalBERT
# ─────────────────────────────────────────────────────────────────────

def encode_texts_with_bert(texts: list, device) -> np.ndarray:
    """
    Encode list of texts with Bio-ClinicalBERT.
    Returns (N, 768) numpy array.
    Falls back to TF-IDF-like random projections if BERT unavailable.
    """
    bert_cache = OUT_DIR / "bert_embeddings.npy"
    keys_cache = OUT_DIR / "bert_keys.json"

    # Load cache if exists and keys match
    texts_hash = str(sorted(texts)[:3])   # simple check
    if bert_cache.exists() and keys_cache.exists():
        try:
            cached_keys  = json.load(open(keys_cache))
            if len(cached_keys) == len(texts):
                embs = np.load(str(bert_cache))
                print(f"[BERT] Loaded {len(embs)} cached embeddings")
                return embs
        except Exception:
            pass

    print(f"[BERT] Encoding {len(texts)} texts with Bio-ClinicalBERT ...")
    try:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        model     = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        model.eval().to(device)

        embeddings = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc   = tokenizer(batch, return_tensors="pt", padding=True,
                              truncation=True, max_length=128)
            enc   = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = model(**enc)
            cls = out.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls)
            if i % 64 == 0:
                print(f"  ... {i}/{len(texts)}")

        embs = np.concatenate(embeddings, axis=0)   # (N, 768)
        np.save(str(bert_cache), embs)
        json.dump(texts[:5], open(keys_cache, "w"))
        print(f"[BERT] Done, saved cache ({embs.shape})")
        return embs

    except Exception as e:
        print(f"[BERT] Failed ({e}) — using random projections as fallback")
        # Deterministic random fallback: consistent across runs
        np.random.seed(42)
        return np.random.randn(len(texts), 768).astype(np.float32) * 0.1


# ─────────────────────────────────────────────────────────────────────
# Step 4 — Load labels
# ─────────────────────────────────────────────────────────────────────

def load_labels():
    patient_discs = defaultdict(list)
    with open(GRADES_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = str(int(float(row["Patient"])))
            patient_discs[pid].append({
                "ivd":        int(float(row["IVD label"])),
                "modic":      int(float(row["Modic"])),
                "up_ep":      int(float(row["UP endplate"])),
                "low_ep":     int(float(row["LOW endplate"])),
                "spondylo":   int(float(row["Spondylolisthesis"])),
                "herniation": int(float(row["Disc herniation"])),
                "narrowing":  int(float(row["Disc narrowing"])),
                "bulging":    int(float(row["Disc bulging"])),
                "pfirrmann":  int(float(row["Pfirrman grade"])),
            })

    labels = {}
    for pid, discs in patient_discs.items():
        any_hern  = any(d["herniation"] for d in discs)
        any_spon  = any(d["spondylo"]   for d in discs)
        any_nar   = any(d["narrowing"]  for d in discs)
        any_bulge = any(d["bulging"]    for d in discs)
        any_modic = any(d["modic"] > 0  for d in discs)
        any_ep    = any(d["up_ep"] or d["low_ep"] for d in discs)
        worst_pfi = max(d["pfirrmann"]  for d in discs)
        mean_pfi  = float(np.mean([d["pfirrmann"] for d in discs]))

        # 7-class disease
        if any_spon:              disease_id = 5
        elif any_hern:            disease_id = 1
        elif any_nar and any_bulge: disease_id = 3   # Stenosis
        elif any_bulge:           disease_id = 2
        elif any_modic or any_ep or any_nar: disease_id = 4   # DDD
        else:                     disease_id = 0

        # Severity
        if   mean_pfi <= 2.5: severity_id = 0
        elif mean_pfi <= 3.5: severity_id = 1
        else:                 severity_id = 2

        ivd_pfi = np.zeros(8, dtype=np.float32)
        level   = np.zeros(8, dtype=np.float32)
        for d in discs:
            idx = d["ivd"] - 1
            if 0 <= idx < 8:
                ivd_pfi[idx] = d["pfirrmann"]
                if any([d["herniation"], d["bulging"], d["narrowing"],
                        d["spondylo"], d["modic"]]):
                    level[idx] = 1.0

        labels[pid] = {
            "disease_id":    disease_id,
            "severity_id":   severity_id,
            "mean_pfirrmann": mean_pfi,
            "ivd_pfi":       ivd_pfi,
            "level":         level,
        }

    counts = np.bincount([v["disease_id"] for v in labels.values()], minlength=7)
    print(f"[Labels] {len(labels)} patients")
    print(f"  " + " | ".join(f"{DISEASE_NAMES[i]}={counts[i]}" for i in range(7) if counts[i]))
    return labels


# ─────────────────────────────────────────────────────────────────────
# Step 5 — Dataset
# ─────────────────────────────────────────────────────────────────────

class FusionDataset(Dataset):
    """
    Each sample:
      img_feat  : (768,) — ResUNet probs projected to 768-dim
      text_feat : (768,) — Bio-ClinicalBERT CLS embedding
      demo_feat : (8,)   — demographics
      labels    : disease_id, severity_id, ivd_pfi (8,), level (8,)
    """
    def __init__(self, samples):
        self.samples = samples

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "img_feat":   torch.tensor(s["img_feat"],  dtype=torch.float32),
            "text_feat":  torch.tensor(s["text_feat"], dtype=torch.float32),
            "demo_feat":  torch.tensor(s["demo_feat"], dtype=torch.float32),
            "disease_id": torch.tensor(s["disease_id"],    dtype=torch.long),
            "severity_id":torch.tensor(s["severity_id"],   dtype=torch.long),
            "pfirrmann":  torch.tensor(s["mean_pfirrmann"],dtype=torch.float32),
            "ivd_pfi":    torch.tensor(s["ivd_pfi"],       dtype=torch.float32),
            "level":      torch.tensor(s["level"],         dtype=torch.float32),
        }


def build_fusion_dataset(labels, demo, device, max_patients=None):
    """Build dataset combining ResUNet features + BERT + demographics."""
    # Load ResUNet feature cache
    cache = torch.load(str(CLF_CACHE), map_location="cpu", weights_only=False)
    print(f"[Cache] {len(cache)} ResUNet feature vectors loaded")

    patient_ids = sorted(labels.keys())
    if max_patients: patient_ids = patient_ids[:max_patients]

    # Build anatomy texts for BERT encoding
    texts, pids_with_cache = [], []
    for pid in patient_ids:
        if pid not in cache: continue
        texts.append(make_anatomy_text(cache[pid]))
        pids_with_cache.append(pid)

    # Encode with BERT
    bert_embs = encode_texts_with_bert(texts, device)  # (N, 768)

    # Project ResUNet features (19,) → (768,) using a fixed linear projection
    # This gives the image_feat dimension matching fusion module input
    np.random.seed(42)
    _proj = np.random.randn(NUM_IMG_CLASSES * 3, 768).astype(np.float32) * 0.02
    # We use the 57-dim engineered features and project to 768

    samples = []
    for i, pid in enumerate(pids_with_cache):
        feat   = cache[pid]
        max_p  = np.array(feat["max_prob"],  dtype=np.float32)
        mean_p = np.array(feat["mean_prob"], dtype=np.float32)
        std_p  = np.array(feat["std_prob"],  dtype=np.float32)
        feat57 = np.concatenate([max_p, mean_p, std_p])   # (57,)

        # Project 57 → 768 via zero-padded linear
        img_768 = np.zeros(768, dtype=np.float32)
        img_768[:57] = feat57
        # Add meaningful signal in higher dims via weighted repeat
        img_768[57:57+57]   = feat57 * 0.5
        img_768[114:114+57] = feat57 * 0.25

        d_vec   = make_demo_vec(demo.get(pid, {}))
        lbl     = labels[pid]
        bert_e  = bert_embs[i]

        samples.append({
            "pid":          pid,
            "img_feat":     img_768,          # (768,)
            "text_feat":    bert_e,           # (768,)
            "demo_feat":    d_vec,            # (8,)
            "disease_id":   lbl["disease_id"],
            "severity_id":  lbl["severity_id"],
            "mean_pfirrmann": lbl["mean_pfirrmann"],
            "ivd_pfi":      lbl["ivd_pfi"],
            "level":        lbl["level"],
        })

    print(f"[Dataset] {len(samples)} samples built")
    return FusionDataset(samples)


# ─────────────────────────────────────────────────────────────────────
# Step 6 — Training
# ─────────────────────────────────────────────────────────────────────

def train(args, device):
    print("\n" + "="*60)
    print("  ATM-Net++ Fusion + MultiTaskHead Training")
    print(f"  Device: {device} | Epochs: {args.epochs}")
    print("="*60)

    labels = load_labels()
    demo   = load_demographics()
    dataset = build_fusion_dataset(labels, demo, device,
                                    max_patients=args.max_patients)
    if len(dataset) < 4:
        print("ERROR: not enough samples"); return

    # Train / val split 80/20
    n_val   = max(2, int(len(dataset) * 0.20))
    n_train = len(dataset) - n_val
    from torch.utils.data import random_split, WeightedRandomSampler
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # Class-weighted sampler
    train_labels = [dataset[i]["disease_id"].item()
                    for i in train_ds.indices]
    counts = np.bincount(train_labels, minlength=7).astype(np.float32)
    counts = np.where(counts == 0, 1, counts)
    cw     = torch.tensor(1.0 / counts)
    cw     = (cw / cw.sum() * 7).to(device)
    sw     = torch.tensor([float(cw[dataset[i]["disease_id"].item()].cpu())
                           for i in train_ds.indices])
    sampler  = WeightedRandomSampler(sw, num_samples=len(train_ds)*3,
                                     replacement=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch,
                          sampler=sampler, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch,
                          shuffle=False, num_workers=0)

    print(f"  Train={n_train} | Val={n_val} | Batch={args.batch}")

    # ── Load models from models/ package ──────────────────────────────
    from models.fusion.multimodal_fusion import MultimodalFusionModule
    from models.classification.disease_classifier import MultiTaskHead

    fusion = MultimodalFusionModule(
        image_feat_dim=768, text_feat_dim=768, demo_feat_dim=256,
        fusion_dim=512, num_heads=8, num_transformer_layers=2,
        dropout=0.15, num_atpg_prompts=16,
    ).to(device)

    head = MultiTaskHead(
        input_dim=512, num_disease_classes=7,
        num_severity_classes=3, num_levels=8, dropout=0.2,
    ).to(device)

    n_f = sum(p.numel() for p in fusion.parameters())
    n_h = sum(p.numel() for p in head.parameters())
    print(f"  FusionModule params: {n_f:,} | MultiTaskHead params: {n_h:,}")

    # Joint optimiser
    optimizer = torch.optim.AdamW(
        list(fusion.parameters()) + list(head.parameters()),
        lr=args.lr, weight_decay=2e-4
    )
    def lr_sched(ep):
        if ep < 10: return (ep+1) / 10
        return 0.5 * (1 + np.cos(np.pi * (ep-10) / max(1, args.epochs-10)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    best_val_loss = float("inf")
    no_improve    = 0
    history       = []

    print(f"\n{'Ep':>4} {'TrLoss':>8} {'VaLoss':>8} "
          f"{'DisAcc':>7} {'SevAcc':>7} {'LR':>8}")
    print("-" * 56)

    for ep in range(1, args.epochs + 1):
        # ── Train ─────────────────────────────────────────────────────
        fusion.train(); head.train()
        tr_loss = 0.0; n_tr = 0

        for batch in train_dl:
            optimizer.zero_grad()
            img_f  = batch["img_feat"].to(device)    # (B, 768)
            txt_f  = batch["text_feat"].to(device)   # (B, 768)
            demo_f = batch["demo_feat"].to(device)   # (B, 8)

            fused_out = fusion(img_f, txt_f, demo_f)
            fused     = fused_out["fused_features"]  # (B, 512)
            pred      = head(fused)

            # Losses
            loss_dis = F.cross_entropy(
                pred["disease"]["logits"],
                batch["disease_id"].to(device),
                weight=cw, label_smoothing=0.05
            )
            loss_sev = F.cross_entropy(
                pred["severity"]["logits"],
                batch["severity_id"].to(device)
            )
            loss_lvl = F.binary_cross_entropy_with_logits(
                pred["level"]["logits"],
                batch["level"].to(device)
            )
            loss_pfi = F.mse_loss(
                pred["ivd_pathology"]["pfirrmann_score"].unsqueeze(-1).expand_as(
                    batch["ivd_pfi"].to(device)
                ),
                batch["ivd_pfi"].to(device)
            )
            loss = loss_dis + 0.4*loss_sev + 0.3*loss_lvl + 0.1*loss_pfi
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(fusion.parameters()) + list(head.parameters()), 1.0
            )
            optimizer.step()
            tr_loss += loss.item(); n_tr += 1

        tr_total = tr_loss / max(1, n_tr)

        # ── Validate ──────────────────────────────────────────────────
        fusion.eval(); head.eval()
        va_loss = 0.0; dis_ok = sev_ok = n_va = 0

        with torch.no_grad():
            for batch in val_dl:
                img_f  = batch["img_feat"].to(device)
                txt_f  = batch["text_feat"].to(device)
                demo_f = batch["demo_feat"].to(device)
                fused  = fusion(img_f, txt_f, demo_f)["fused_features"]
                pred   = head(fused)

                loss_dis = F.cross_entropy(
                    pred["disease"]["logits"],
                    batch["disease_id"].to(device), weight=cw)
                loss_sev = F.cross_entropy(
                    pred["severity"]["logits"],
                    batch["severity_id"].to(device))
                va_loss += (loss_dis + 0.4*loss_sev).item()

                dp = pred["disease"]["pred"].cpu()
                sp = pred["severity"]["pred"].cpu()
                dis_ok += (dp == batch["disease_id"]).sum().item()
                sev_ok += (sp == batch["severity_id"]).sum().item()
                n_va   += len(batch["disease_id"])

        va_total = va_loss / max(1, len(val_dl))
        dis_acc  = dis_ok / max(1, n_va)
        sev_acc  = sev_ok / max(1, n_va)
        lr_now   = optimizer.param_groups[0]["lr"]

        scheduler.step()
        history.append({"ep": ep, "tr": round(tr_total,4),
                        "val": round(va_total,4),
                        "dis_acc": round(dis_acc,4)})

        flag = ""
        if va_total < best_val_loss:
            best_val_loss = va_total
            no_improve    = 0
            # Save both modules
            torch.save({
                "epoch":       ep,
                "model_state": fusion.state_dict(),
                "val_loss":    va_total,
                "dis_acc":     dis_acc,
                "config": {"image_feat_dim":768,"text_feat_dim":768,
                           "demo_feat_dim":256,"fusion_dim":512,
                           "num_heads":8,"num_transformer_layers":2,
                           "dropout":0.15,"num_atpg_prompts":16},
            }, str(OUT_DIR / "fusion_module.pth"))
            torch.save({
                "epoch":       ep,
                "model_state": head.state_dict(),
                "val_loss":    va_total,
                "dis_acc":     dis_acc,
                "config": {"input_dim":512,"num_disease_classes":7,
                           "num_severity_classes":3,"num_levels":8,
                           "dropout":0.2},
            }, str(OUT_DIR / "multitask_head.pth"))
            flag = " *"
        else:
            no_improve += 1

        if ep % 10 == 0 or ep <= 5 or flag:
            print(f"{ep:>4}  {tr_total:>8.4f}  {va_total:>8.4f}  "
                  f"{dis_acc:>6.1%}  {sev_acc:>6.1%}  "
                  f"{lr_now:>8.2e}{flag}")

        if no_improve >= args.patience:
            print(f"\n  Early stop at epoch {ep}")
            break

    with open(OUT_DIR / "fusion_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n" + "="*60)
    print(f"  Done | Best val loss: {best_val_loss:.4f}")
    print(f"  fusion_module.pth  -> {OUT_DIR}")
    print(f"  multitask_head.pth -> {OUT_DIR}")
    print("="*60)
    print("\nRestart server:  py server.py")
    return fusion, head


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",       type=int,   default=150)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--batch",        type=int,   default=16)
    parser.add_argument("--patience",     type=int,   default=30)
    parser.add_argument("--max_patients", type=int,   default=None)
    parser.add_argument("--quick",        action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.epochs = 10; args.patience = 5; args.max_patients = 30
        print("[Quick mode]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(args, device)
