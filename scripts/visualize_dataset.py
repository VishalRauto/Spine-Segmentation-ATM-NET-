"""
Dataset visualization script.
Renders sample MRI slices with ground-truth masks.

Usage:
    python scripts/visualize_dataset.py --n-samples 20 --output outputs/viz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",    default="configs/base_config.yaml")
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--output",    default="outputs/dataset_viz")
    p.add_argument("--modality",  default="t2", choices=["t1", "t2"])
    p.add_argument("--split",     default="training", choices=["training", "validation"])
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    from datasets.preprocessing.mha_reader import MedicalImageReader
    from datasets.preprocessing.label_mapper import (
        remap_spider_mask, create_colorized_mask,
        ATMNET_COLORMAP, ATMNET_TO_NAME
    )

    paths = config.get("paths", {})
    images_dir = Path(paths.get("images_dir", "data/10159290/images"))
    masks_dir  = Path(paths.get("masks_dir",  "data/10159290/masks"))
    overview_csv = paths.get("overview_csv", "data/10159290/overview.csv")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get patient IDs for split
    df = pd.read_csv(overview_csv)
    split_df = df[df["new_file_name"].str.endswith(f"_{args.modality}")]
    if "subset" in df.columns:
        split_df = split_df[split_df["subset"] == args.split]
    patient_ids = [r.replace(f"_{args.modality}", "")
                   for r in split_df["new_file_name"].tolist()]
    patient_ids = patient_ids[:args.n_samples]

    if not patient_ids:
        print(f"No patients found for split={args.split}, modality={args.modality}")
        return

    reader = MedicalImageReader()
    count = 0

    for pid in patient_ids:
        fname = f"{pid}_{args.modality}.mha"
        img_path = images_dir / fname
        msk_path = masks_dir / fname

        if not img_path.exists() or not msk_path.exists():
            continue

        img = reader.read(img_path)
        msk = reader.read(msk_path)

        # Pick middle slice
        n_slices = img.data.shape[0]
        mid = n_slices // 2
        img_slice = img.data[mid].astype(np.float32)
        msk_slice = remap_spider_mask(msk.data[mid].astype(np.int32))

        # Normalize
        p_lo, p_hi = np.percentile(img_slice, [1, 99])
        img_norm = np.clip((img_slice - p_lo) / (p_hi - p_lo + 1e-8), 0, 1)

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Patient {pid} — {args.modality.upper()} — Slice {mid}/{n_slices-1}",
                     fontsize=13, fontweight="bold")

        # Original
        axes[0].imshow(img_norm, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("MRI Input", fontsize=11)
        axes[0].axis("off")

        # Mask
        mask_rgb = create_colorized_mask(msk_slice)
        axes[1].imshow(mask_rgb)
        axes[1].set_title("Ground Truth Segmentation", fontsize=11)
        axes[1].axis("off")

        # Overlay
        overlay = (img_norm[:, :, np.newaxis] * 255).astype(np.uint8)
        overlay = np.repeat(overlay, 3, axis=2)
        nonzero = msk_slice > 0
        overlay[nonzero] = (
            0.45 * overlay[nonzero] +
            0.55 * mask_rgb[nonzero]
        ).astype(np.uint8)
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay", fontsize=11)
        axes[2].axis("off")

        # Legend for present classes
        present = np.unique(msk_slice)
        patches = [
            mpatches.Patch(
                color=[c / 255 for c in ATMNET_COLORMAP.get(c_id, (128,128,128))],
                label=ATMNET_TO_NAME.get(c_id, f"class_{c_id}")
            )
            for c_id in present if c_id > 0
        ]
        if patches:
            fig.legend(handles=patches, loc="lower center", ncol=min(len(patches), 6),
                      fontsize=8, bbox_to_anchor=(0.5, -0.05))

        plt.tight_layout()
        save_path = out_dir / f"{pid}_{args.modality}_slice{mid:03d}.png"
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        count += 1
        print(f"[{count}/{len(patient_ids)}] Saved: {save_path.name}")

    print(f"\n[✓] Saved {count} visualizations to {out_dir}")


if __name__ == "__main__":
    main()
