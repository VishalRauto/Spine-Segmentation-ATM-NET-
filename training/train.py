"""
Main training script for ATM-Net++.

Usage:
    python training/train.py --config configs/base_config.yaml
    python training/train.py --config configs/base_config.yaml --resume checkpoints/best.pth
    python training/train.py --config configs/base_config.yaml --fold 0  # K-Fold
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/training.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_model(config: dict):
    from models.atmnet_plus_plus import ATMNetPlusPlus
    model_cfg = config.get("model", {})
    seg_cfg = config.get("segmentation", {})
    cls_cfg = config.get("classification", {})
    fusion_cfg = config.get("fusion", {})

    model = ATMNetPlusPlus(
        img_size=tuple(model_cfg.get("img_size", [512, 512])),
        in_channels=model_cfg.get("in_channels", 1),
        num_seg_classes=seg_cfg.get("num_classes", 20),
        num_disease_classes=cls_cfg.get("num_disease_classes", 7),
        feature_size=model_cfg.get("feature_size", 48),
        fusion_dim=fusion_cfg.get("fusion_dim", 512),
        text_model_name=config.get("text_encoder", {}).get("model_name", "emilyalsentzer/Bio_ClinicalBERT"),
        deep_supervision=True,
        use_text=True,
        use_demographics=True,
        dropout=0.1,
    )
    return model


def build_loss(config: dict):
    from training.losses.combined_loss import ATMNetLoss
    loss_cfg = config.get("training", {}).get("loss", {})
    return ATMNetLoss(
        num_seg_classes=config.get("segmentation", {}).get("num_classes", 20),
        num_disease_classes=config.get("classification", {}).get("num_disease_classes", 7),
        seg_dice_weight=float(loss_cfg.get("seg_dice_weight", 1.0)),
        seg_focal_weight=float(loss_cfg.get("seg_focal_weight", 0.5)),
        seg_boundary_weight=float(loss_cfg.get("seg_boundary_weight", 0.2)),
        cls_weight=float(loss_cfg.get("cls_weight", 0.3)),
        severity_weight=float(loss_cfg.get("severity_weight", 0.2)),
        level_weight=float(loss_cfg.get("level_weight", 0.2)),
        report_weight=float(loss_cfg.get("report_weight", 0.1)),
        contrastive_weight=0.05,
        ds_weight=0.4,
    )


def build_transforms(config: dict):
    from datasets.transforms.augmentations import SpineAugmentor
    aug_cfg = config.get("augmentation", {})

    train_transform = SpineAugmentor(
        rotation_range=float(aug_cfg.get("rotation_range", 15)),
        flip_prob=float(aug_cfg.get("flip_prob", 0.5)),
        elastic_alpha=float(aug_cfg.get("elastic_alpha", 100)),
        elastic_sigma=float(aug_cfg.get("elastic_sigma", 10)),
        intensity_shift=float(aug_cfg.get("intensity_shift", 0.1)),
        intensity_scale=float(aug_cfg.get("intensity_scale", 0.1)),
        gaussian_noise_std=float(aug_cfg.get("gaussian_noise_std", 0.01)),
        random_crop_size=tuple(aug_cfg.get("random_crop_size", [480, 480])),
        zoom_range=tuple(aug_cfg.get("zoom_range", [0.9, 1.1])),
        p_elastic=0.3,
        p_gamma=0.3,
        p_noise=0.5,
        p_intensity_shift=0.5,
        p_zoom=0.3,
        training=True,
    )
    val_transform = SpineAugmentor(training=False)
    return train_transform, val_transform


def build_preprocessor(config: dict):
    from datasets.preprocessing.normalizer import SpinePreprocessor, NormalizationStrategy
    data_cfg = config.get("data", {})
    return SpinePreprocessor(
        target_size=tuple(data_cfg.get("image_size", [512, 512])),
        normalize_strategy=NormalizationStrategy.PERCENTILE,
        add_channel_dim=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Train ATM-Net++")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--fold", type=int, default=None, help="K-Fold fold index (0-based)")
    parser.add_argument("--experiment", type=str, default="atmnet_pp")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--debug", action="store_true", help="Quick debug run")
    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).parent.parent / args.config
    config = load_config(str(config_path))

    # Seed
    seed = config.get("training", {}).get("seed", 42)
    seed_everything(seed)

    # Debug mode overrides
    if args.debug:
        config["training"]["epochs"] = 3
        config["training"]["batch_size"] = 2
        config["training"]["num_workers"] = 0
        logger.info("DEBUG MODE: Reduced epochs and batch size")

    # Build data paths
    paths = config.get("paths", {})
    images_dir = paths.get("images_dir", "data/10159290/images")
    masks_dir = paths.get("masks_dir", "data/10159290/masks")
    overview_csv = paths.get("overview_csv", "data/10159290/overview.csv")
    gradings_csv = paths.get("gradings_csv", "data/10159290/radiological_gradings.csv")

    # Build components
    logger.info("Building model...")
    model = build_model(config)
    logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    loss_fn = build_loss(config)
    train_transform, val_transform = build_transforms(config)
    preprocessor = build_preprocessor(config)

    # Build DataLoaders
    logger.info("Building DataLoaders...")
    from datasets.loaders.spider_dataset import create_dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        config=config,
        images_dir=images_dir,
        masks_dir=masks_dir,
        overview_csv=overview_csv,
        gradings_csv=gradings_csv,
        transform_train=train_transform,
        transform_val=val_transform,
        preprocessor=preprocessor,
    )
    logger.info(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")

    # Build Trainer
    from training.trainer import Trainer
    output_dir = paths.get("checkpoints_dir", "checkpoints")
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        config=config,
        output_dir=output_dir,
        experiment_name=args.experiment or "atmnet_pp",
        use_wandb=False,
    )

    # Train
    result = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        resume_from=args.resume,
    )

    logger.info(f"Training complete. Best Dice: {result['best_dice']:.4f}")

    # Final evaluation on test set
    logger.info("Running final evaluation on test set...")
    from evaluation.evaluator import Evaluator
    evaluator = Evaluator(model=model, device=trainer.device, config=config)
    test_metrics = evaluator.evaluate(test_loader)
    logger.info(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    # Create log dir
    os.makedirs("logs", exist_ok=True)
    main()
