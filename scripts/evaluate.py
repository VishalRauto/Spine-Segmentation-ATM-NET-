"""
Full evaluation script for ATM-Net++.
Evaluates on the SPIDER test set and saves comprehensive metrics.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/atmnet_pp_best.pth
    python scripts/evaluate.py --checkpoint checkpoints/best.pth --save-predictions
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",       default="checkpoints/atmnet_pp_best.pth")
    p.add_argument("--config",           default="configs/base_config.yaml")
    p.add_argument("--output-dir",       default="outputs/evaluation")
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--compute-hd",       action="store_true", help="Compute HD95 (slow)")
    p.add_argument("--tta",              action="store_true")
    p.add_argument("--device",           default="auto")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    paths = config.get("paths", {})

    # Build model
    logger.info("Loading model...")
    from models.atmnet_plus_plus import ATMNetPlusPlus
    model_cfg = config.get("model", {})
    model = ATMNetPlusPlus(
        img_size=tuple(model_cfg.get("img_size", [512, 512])),
        in_channels=model_cfg.get("in_channels", 1),
        num_seg_classes=config.get("segmentation", {}).get("num_classes", 20),
        use_text=True,
        use_demographics=True,
        deep_supervision=False,
    )

    if Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Checkpoint epoch: {ckpt.get('epoch','?')}, best dice: {ckpt.get('best_dice','?'):.4f}")
    else:
        logger.warning(f"No checkpoint found at {args.checkpoint}. Evaluating untrained model.")

    model = model.to(device)

    # Build test DataLoader
    logger.info("Building test dataset...")
    from datasets.preprocessing.normalizer import SpinePreprocessor
    from training.train import build_transforms
    _, val_transform = build_transforms(config)
    preprocessor = SpinePreprocessor(
        target_size=tuple(config.get("data", {}).get("image_size", [512, 512]))
    )

    from datasets.loaders.spider_dataset import create_dataloaders
    _, _, test_loader = create_dataloaders(
        config=config,
        images_dir=paths.get("images_dir", "data/10159290/images"),
        masks_dir=paths.get("masks_dir", "data/10159290/masks"),
        overview_csv=paths.get("overview_csv", "data/10159290/overview.csv"),
        gradings_csv=paths.get("gradings_csv", "data/10159290/radiological_gradings.csv"),
        preprocessor=preprocessor,
    )

    # Evaluate
    from evaluation.evaluator import Evaluator
    evaluator = Evaluator(model=model, device=device, config=config, output_dir=args.output_dir)
    metrics = evaluator.evaluate(
        test_loader,
        save_results=True,
        compute_hd=args.compute_hd,
        tta=args.tta,
    )

    # Save predictions
    if args.save_predictions:
        pred_dir = Path(args.output_dir) / "predictions"
        logger.info(f"Saving prediction images to {pred_dir}")
        evaluator.export_predictions(test_loader, str(pred_dir))

    # Print final table
    print("\n" + "=" * 50)
    print("  FINAL EVALUATION METRICS")
    print("=" * 50)
    key_metrics = ["mean_dice", "mean_iou", "macro_f1", "mean_hd95",
                   "disease_accuracy", "severity_accuracy"]
    for k in key_metrics:
        if k in metrics:
            v = metrics[k]
            star = " ✅" if k == "mean_dice" and v > 0.90 else ""
            print(f"  {k:35s}: {v:.4f}{star}")
    print("=" * 50)

    # Check Dice target
    if metrics.get("mean_dice", 0) > 0.90:
        print("\n  🎯 TARGET ACHIEVED: Dice > 0.90")
    else:
        print(f"\n  📊 Current Dice: {metrics.get('mean_dice',0):.4f} (Target: > 0.90)")


if __name__ == "__main__":
    main()
