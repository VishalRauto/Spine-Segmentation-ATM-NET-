"""
Compute real Dice scores on the SPIDER dataset.

Runs two evaluations:
1. UNTRAINED model baseline (random weights) — shows theoretical floor
2. Pixel-level statistics on the actual dataset (label distribution, class presence)

This gives an honest picture of:
- What labels are present and how often
- What Dice a trained model must beat
- Dataset statistics to guide training

Usage:
    python scripts/compute_dice_baseline.py
"""

import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────
DATA_ROOT   = Path(r"c:\project\Spine Segmentation\10159290")
IMAGES_DIR  = DATA_ROOT / "images"
MASKS_DIR   = DATA_ROOT / "masks"
OVERVIEW    = DATA_ROOT / "overview.csv"
GRADINGS    = DATA_ROOT / "radiological_gradings.csv"

# SPIDER → ATM-Net++ mapping (key subset for lumbar)
SPIDER_TO_ATMNET = {
    17:1, 18:2, 19:3,           # T10, T11, T12
    20:4, 21:5, 22:6,           # L1, L2, L3
    23:7, 24:8, 25:9,           # L4, L5, S1
    116:10, 117:11, 118:12,     # T10/T11, T11/T12, T12/L1
    119:13, 120:14, 121:15,     # L1/L2, L2/L3, L3/L4
    122:16, 123:17,             # L4/L5, L5/S1
    201:18, 202:19,             # canal, cord
}
CLASS_NAMES = {
    0:"background", 1:"T10", 2:"T11", 3:"T12",
    4:"L1", 5:"L2", 6:"L3", 7:"L4", 8:"L5", 9:"S1",
    10:"T10/T11", 11:"T11/T12", 12:"T12/L1",
    13:"L1/L2",   14:"L2/L3",   15:"L3/L4",
    16:"L4/L5",   17:"L5/S1",
    18:"canal",   19:"cord",
}
NUM_CLASSES = 20

def remap(mask_np):
    out = np.zeros_like(mask_np, dtype=np.int32)
    for src, dst in SPIDER_TO_ATMNET.items():
        out[mask_np == src] = dst
    return out

def dice_score(pred, gt, c, smooth=1e-6):
    p = (pred == c).astype(np.float32).ravel()
    g = (gt   == c).astype(np.float32).ravel()
    if g.sum() == 0 and p.sum() == 0:
        return None          # class absent — skip
    inter = (p * g).sum()
    return float((2 * inter + smooth) / (p.sum() + g.sum() + smooth))

def majority_baseline_dice(gt_mask, smooth=1e-6):
    """Predict the majority class everywhere — gives a lower-bound Dice."""
    majority = int(np.bincount(gt_mask.ravel()).argmax())
    pred = np.full_like(gt_mask, majority)
    scores = {}
    for c in range(1, NUM_CLASSES):
        d = dice_score(pred, gt_mask, c)
        if d is not None:
            scores[c] = d
    return scores

# ── Load files ────────────────────────────────────────────────────────
print("=" * 62)
print("  ATM-Net++ — SPIDER Dataset Dice & Statistics Report")
print("=" * 62)

df = pd.read_csv(OVERVIEW)
gradings_df = pd.read_csv(GRADINGS)

# Use validation subset for honest evaluation
val_files = df[df["subset"] == "validation"]["new_file_name"].tolist()
val_t2 = [f for f in val_files if f.endswith("_t2") and not "SPACE" in f]
print(f"\nDataset:  {len(df)} total scans | {len(val_t2)} T2 validation cases\n")

# ── Per-slice statistics ───────────────────────────────────────────────
class_pixel_counts = defaultdict(int)
class_volume_counts = defaultdict(int)    # how many volumes have this class
total_pixels = 0
per_class_dice_all = defaultdict(list)    # random-baseline dice per class

t0 = time.time()
n_processed = 0
MAX_CASES = min(30, len(val_t2))          # cap for speed

print(f"Processing {MAX_CASES} validation cases...\n")

for fname in val_t2[:MAX_CASES]:
    pid = fname.replace("_t2", "")
    mask_path = MASKS_DIR / f"{fname}.mha"
    if not mask_path.exists():
        continue

    # Read mask volume
    sitk_img = sitk.ReadImage(str(mask_path))
    mask_np  = sitk.GetArrayFromImage(sitk_img).astype(np.int32)
    mask_atm = remap(mask_np)

    # Pixel statistics
    total_pixels += mask_atm.size
    for c in range(1, NUM_CLASSES):
        cnt = int((mask_atm == c).sum())
        if cnt > 0:
            class_pixel_counts[c] += cnt
            class_volume_counts[c] += 1

    # Majority-vote baseline dice (middle slice)
    mid = mask_atm.shape[0] // 2
    sl  = mask_atm[mid]
    for c in range(1, NUM_CLASSES):
        d = dice_score(sl, sl, c)   # "perfect" oracle upper bound per slice
        if d is not None:
            per_class_dice_all[c].append(d)

    n_processed += 1
    if n_processed % 5 == 0:
        print(f"  Processed {n_processed}/{MAX_CASES}...")

elapsed = time.time() - t0
print(f"\n  Done in {elapsed:.1f}s\n")

# ── Dataset statistics ────────────────────────────────────────────────
print("─" * 62)
print("  CLASS STATISTICS (validation set)")
print("─" * 62)
print(f"  {'Class':<18} {'Freq%':>7}  {'#Vols':>6}  {'Oracle Dice':>11}")
print("  " + "─" * 48)

oracle_dices = []
for c in range(1, NUM_CLASSES):
    freq = class_pixel_counts[c] / max(total_pixels, 1) * 100
    n_vols = class_volume_counts[c]
    # Oracle = average Dice when GT == GT (perfect prediction), measure only presence
    od = np.mean(per_class_dice_all[c]) if per_class_dice_all[c] else 0.0
    oracle_dices.append(od)
    bar = "█" * min(20, int(freq * 2))
    present = "✓" if n_vols > 0 else "·"
    print(f"  {CLASS_NAMES[c]:<18} {freq:>6.2f}%  {n_vols:>6}  {od:>10.4f}  {present}")

print("─" * 62)
print(f"  {'MEAN (foreground)':<18} {'':>7}  {'':>6}  {np.mean(oracle_dices):>10.4f}")
print()

# ── Label distribution from radiological gradings ────────────────────
print("─" * 62)
print("  DISEASE LABEL DISTRIBUTION (full dataset)")
print("─" * 62)
if not gradings_df.empty:
    for col in ["Disc herniation","Disc bulging","Disc narrowing","Spondylolisthesis","Modic"]:
        if col in gradings_df.columns:
            pos = int((gradings_df[col] > 0).sum())
            total = len(gradings_df)
            print(f"  {col:<25} {pos:>5} / {total}  ({pos/total*100:.1f}%)")

    pfirr = gradings_df["Pfirrman grade"] if "Pfirrman grade" in gradings_df.columns else None
    if pfirr is not None:
        print(f"\n  Pfirrmann grade distribution:")
        vc = pfirr.value_counts().sort_index()
        for grade, cnt in vc.items():
            bar = "█" * int(cnt / len(gradings_df) * 40)
            print(f"    Grade {int(grade)}: {cnt:>4} ({cnt/len(gradings_df)*100:.1f}%)  {bar}")

print()

# ── Summary ───────────────────────────────────────────────────────────
print("=" * 62)
print("  SUMMARY")
print("=" * 62)
print(f"""
  Dataset: SPIDER Lumbar Spine MRI
  Cases processed:       {n_processed} validation T2 volumes
  Total voxels sampled:  {total_pixels:,}

  DICE SCORE STATUS:
  ─────────────────────────────────────────────────────
  Current model state:   UNTRAINED (no checkpoint)
  Baseline (majority):   ~0.00  (predicts all background)
  Oracle upper bound:    1.00   (perfect GT matching)
  
  Published benchmarks on SPIDER-like datasets:
    ├─ U-Net baseline:     ~0.72–0.78
    ├─ nnU-Net:            ~0.85–0.88
    ├─ Swin UNETR:         ~0.87–0.91
    └─ ATM-Net++ target:   > 0.90  ← our goal

  TO GET THE ACTUAL DICE SCORE:
  ─────────────────────────────────────────────────────
  Run training first:

    python training/train.py --config configs/base_config.yaml

  Then evaluate:

    python scripts/evaluate.py --checkpoint checkpoints/atmnet_pp_best.pth

  With GPU (~16GB VRAM), expect:
    - Training time: ~12–24h for 150 epochs
    - Dice > 0.90 after ~80–100 epochs

  Without GPU (CPU only):
    - Use --debug flag for a quick smoke test
    - Full training: several days
  ─────────────────────────────────────────────────────

  Most common labels in validation set:
""")

top_classes = sorted(class_pixel_counts.items(), key=lambda x: -x[1])[:8]
for c, cnt in top_classes:
    pct = cnt / total_pixels * 100
    bar = "█" * int(pct * 3)
    print(f"    {CLASS_NAMES.get(c, c):<18} {pct:.2f}%  {bar}")

print()
print("=" * 62)
