"""Inspect actual label values in SPIDER masks and compute real Dice."""

import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from collections import Counter

DATA_ROOT  = Path(r"c:\project\Spine Segmentation\10159290")
MASKS_DIR  = DATA_ROOT / "masks"
OVERVIEW   = DATA_ROOT / "overview.csv"

df = pd.read_csv(OVERVIEW)
val_files = df[df["subset"] == "validation"]["new_file_name"].tolist()
val_t2 = [f for f in val_files if f.endswith("_t2") and "SPACE" not in f][:5]

print("=" * 60)
print("  Actual label values in SPIDER masks")
print("=" * 60)

all_labels = Counter()

for fname in val_t2:
    path = MASKS_DIR / f"{fname}.mha"
    if not path.exists():
        continue
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).ravel()
    unique, counts = np.unique(arr, return_counts=True)
    print(f"\n{fname}:")
    for u, c in zip(unique, counts):
        pct = c / arr.size * 100
        if pct > 0.001:
            print(f"  label {u:>5}  →  {c:>8} px  ({pct:.3f}%)")
            all_labels[int(u)] += int(c)

print("\n" + "=" * 60)
print("  Aggregate label frequencies across 5 cases")
print("=" * 60)
total = sum(all_labels.values())
for lbl, cnt in sorted(all_labels.items(), key=lambda x: -x[1]):
    if cnt / total > 0.0001:
        print(f"  Label {lbl:>5}  :  {cnt:>9,} px  ({cnt/total*100:.3f}%)")

# Show num_vertebrae and num_discs from overview
print("\n" + "=" * 60)
print("  Overview stats for validation cases")
print("=" * 60)
vdf = df[df["new_file_name"].isin(val_t2)][["new_file_name","num_vertebrae","num_discs","sex"]]
print(vdf.to_string(index=False))
