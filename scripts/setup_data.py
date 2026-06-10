"""
Data setup script: creates symlinks or copies from the source SPIDER dataset
into the expected project data directory structure.

Usage:
    python scripts/setup_data.py
    python scripts/setup_data.py --source "C:/project/Spine Segmentation/10159290"
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source",
        default=r"C:\project\Spine Segmentation\10159290",
        help="Path to the SPIDER dataset root (contains images/, masks/, overview.csv)",
    )
    p.add_argument("--config", default="configs/base_config.yaml")
    p.add_argument("--copy", action="store_true", help="Copy files instead of symlink")
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.source)

    if not src.exists():
        print(f"[ERROR] Source not found: {src}")
        sys.exit(1)

    # Verify expected contents
    required = ["images", "masks", "overview.csv", "radiological_gradings.csv"]
    for r in required:
        if not (src / r).exists():
            print(f"[ERROR] Missing: {src / r}")
            sys.exit(1)

    # Load config to get target paths
    with open(args.config) as f:
        config = yaml.safe_load(f)

    paths = config.get("paths", {})
    data_root = Path(paths.get("data_root", "data/10159290"))
    data_root.mkdir(parents=True, exist_ok=True)

    print(f"[Setup] Source: {src}")
    print(f"[Setup] Target: {data_root.absolute()}")

    items = [
        ("images",                   data_root / "images"),
        ("masks",                    data_root / "masks"),
        ("overview.csv",             data_root / "overview.csv"),
        ("radiological_gradings.csv",data_root / "radiological_gradings.csv"),
    ]

    for src_name, dst_path in items:
        src_path = src / src_name
        if dst_path.exists() or dst_path.is_symlink():
            print(f"[Skip] Already exists: {dst_path.name}")
            continue

        if args.copy:
            if src_path.is_dir():
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)
            print(f"[Copy] {src_name} -> {dst_path}")
        else:
            # Symlink (faster, no disk duplication)
            try:
                os.symlink(src_path.resolve(), dst_path)
                print(f"[Link] {src_name} -> {dst_path}")
            except OSError as e:
                print(f"[Warning] Symlink failed ({e}). Copying instead.")
                if src_path.is_dir():
                    shutil.copytree(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)
                print(f"[Copy] {src_name} -> {dst_path}")

    # Count files
    img_dir = data_root / "images"
    msk_dir = data_root / "masks"
    if img_dir.exists():
        n_img = len(list(img_dir.glob("*.mha")))
        n_msk = len(list(msk_dir.glob("*.mha"))) if msk_dir.exists() else 0
        print(f"\n[✓] Found {n_img} images, {n_msk} masks")

    print("\n[✓] Data setup complete. Ready to train!")
    print(f"    Run: python training/train.py --config {args.config}")


if __name__ == "__main__":
    main()
