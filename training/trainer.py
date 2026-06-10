"""
ATM-Net++ Training Engine.

Features:
- Mixed precision training (torch.cuda.amp)
- Gradient accumulation
- Automatic checkpointing
- Early stopping
- TensorBoard + W&B logging
- K-Fold cross-validation support
- LR scheduling (cosine warmup)
- Resume from checkpoint
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class WarmupCosineScheduler:
    """Cosine LR schedule with linear warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            # Linear warmup
            factor = (epoch + 1) / max(self.warmup_epochs, 1)
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / max(self.max_epochs - self.warmup_epochs, 1)
            factor = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
            factor = max(factor, self.min_lr / self.base_lrs[0])

        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor

    def get_last_lr(self) -> List[float]:
        return [group["lr"] for group in self.optimizer.param_groups]


class EarlyStopping:
    """Early stopping with patience and minimum delta."""

    def __init__(self, patience: int = 30, min_delta: float = 0.001, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        if self.mode == "max":
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement.")

        return self.should_stop


class Trainer:
    """
    Complete training engine for ATM-Net++.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: dict,
        output_dir: str = "checkpoints",
        experiment_name: str = "atmnet_pp",
        use_wandb: bool = False,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_dice = 0.0
        self.history: List[Dict] = []

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training on device: {self.device}")
        self.model = self.model.to(self.device)

        # Mixed precision
        train_cfg = config.get("training", {})
        self.use_amp = train_cfg.get("mixed_precision", True) and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        # Gradient accumulation
        self.accumulation_steps = train_cfg.get("accumulation_steps", 4)

        # Optimizer
        opt_cfg = train_cfg.get("optimizer", {})
        self.optimizer = AdamW(
            model.parameters(),
            lr=float(opt_cfg.get("lr", 1e-4)),
            weight_decay=float(opt_cfg.get("weight_decay", 1e-5)),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )

        # LR Scheduler
        sched_cfg = train_cfg.get("scheduler", {})
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=sched_cfg.get("warmup_epochs", 10),
            max_epochs=train_cfg.get("epochs", 200),
            min_lr=float(sched_cfg.get("min_lr", 1e-6)),
        )

        # Early stopping
        es_cfg = train_cfg.get("early_stopping", {})
        self.early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 30),
            min_delta=es_cfg.get("min_delta", 0.001),
            mode="max",
        )

        # TensorBoard
        tb_dir = str(self.output_dir / "tensorboard" / experiment_name)
        self.writer = SummaryWriter(log_dir=tb_dir)

        # W&B
        if use_wandb:
            self._init_wandb()

        # Metric tracker
        from training.metrics.segmentation_metrics import SegmentationMetricTracker
        num_classes = config.get("segmentation", {}).get("num_classes", 20)
        class_names = config.get("segmentation", {}).get("class_names", None)
        self.metric_tracker = SegmentationMetricTracker(num_classes, class_names)

    def _init_wandb(self):
        try:
            import wandb
            wandb_cfg = self.config.get("wandb", {})
            wandb.init(
                project=wandb_cfg.get("project", "atmnet-plus-plus"),
                name=self.experiment_name,
                config=self.config,
            )
            self._wandb = wandb
        except ImportError:
            logger.warning("wandb not installed. Skipping W&B logging.")
            self.use_wandb = False
            self._wandb = None

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None,
        resume_from: Optional[str] = None,
    ) -> Dict:
        """
        Main training loop.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            num_epochs: Override config epochs
            resume_from: Path to checkpoint to resume from

        Returns:
            Training history dict
        """
        if resume_from:
            self._load_checkpoint(resume_from)

        max_epochs = num_epochs or self.config.get("training", {}).get("epochs", 200)

        logger.info(f"Starting training for {max_epochs} epochs")
        logger.info(f"Model params: {self.model.get_num_parameters() if hasattr(self.model, 'get_num_parameters') else 'N/A'}")

        for epoch in range(self.epoch, max_epochs):
            self.epoch = epoch
            self.scheduler.step(epoch)
            current_lr = self.scheduler.get_last_lr()[0]

            # Training epoch
            train_metrics = self._train_epoch(train_loader)

            # Validation epoch
            val_metrics = self._val_epoch(val_loader)

            # Logging
            self._log_epoch(epoch, train_metrics, val_metrics, current_lr)

            # Save best checkpoint
            val_dice = val_metrics.get("mean_dice", 0.0)
            if val_dice > self.best_dice:
                self.best_dice = val_dice
                self._save_checkpoint(epoch, val_metrics, is_best=True)
                logger.info(f"[Epoch {epoch+1}] New best Dice: {val_dice:.4f}")

            # Periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, val_metrics, is_best=False)

            # Record history
            self.history.append({
                "epoch": epoch + 1,
                "lr": current_lr,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            })

            # Early stopping
            if self.early_stopping(val_dice):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        self.writer.close()
        self._save_history()
        return {"best_dice": self.best_dice, "history": self.history}

    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        self.metric_tracker.reset()
        epoch_losses: Dict[str, List[float]] = {}
        step = 0

        for batch_idx, batch in enumerate(loader):
            # Move batch to device
            batch = self._batch_to_device(batch)

            # Build model inputs
            model_inputs = self._build_model_inputs(batch)

            with autocast(enabled=self.use_amp):
                output = self.model(**model_inputs, return_deep_supervision=True)
                losses = self.loss_fn(output, batch)
                loss = losses["total"] / self.accumulation_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(loader):
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Record losses
            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    val = v.item()
                    epoch_losses.setdefault(k, []).append(val)

            # Update metrics
            with torch.no_grad():
                self.metric_tracker.update(
                    output["seg_logits"].detach(),
                    batch["mask"],
                    compute_hd=False,
                )

            step += 1

        # Aggregate
        avg_losses = {k: float(sum(v) / len(v)) for k, v in epoch_losses.items()}
        seg_metrics = self.metric_tracker.compute()
        return {**avg_losses, **seg_metrics}

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Run one validation epoch."""
        self.model.eval()
        self.metric_tracker.reset()
        epoch_losses: Dict[str, List[float]] = {}

        for batch in loader:
            batch = self._batch_to_device(batch)
            model_inputs = self._build_model_inputs(batch)

            with autocast(enabled=self.use_amp):
                output = self.model(**model_inputs)
                losses = self.loss_fn(output, batch)

            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    epoch_losses.setdefault(k, []).append(v.item())

            self.metric_tracker.update(
                output["seg_logits"],
                batch["mask"],
                compute_hd=False,  # Enable HD for final eval
            )

        avg_losses = {k: float(sum(v) / len(v)) for k, v in epoch_losses.items()}
        seg_metrics = self.metric_tracker.compute()
        return {**avg_losses, **seg_metrics}

    def _build_model_inputs(self, batch: Dict) -> Dict:
        """Build model input dict from batch."""
        inputs = {"image": batch["image"]}

        if "report_text" in batch and hasattr(self.model, "text_encoder") and self.model.text_encoder is not None:
            texts = list(batch["report_text"])
            tokenizer = self.model.text_encoder.tokenizer
            if tokenizer is not None:
                try:
                    encoding = self.model.text_encoder.tokenize(texts, device=self.device)
                    inputs.update(encoding)
                except Exception:
                    pass

        if "demographics" in batch:
            inputs["demographics"] = batch["demographics"]

        return inputs

    def _batch_to_device(self, batch: Dict) -> Dict:
        """Move tensor values in batch to device."""
        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device, non_blocking=True)
            else:
                result[k] = v
        return result

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict,
        lr: float,
    ):
        """Log metrics to TensorBoard and W&B."""
        # TensorBoard
        for k, v in train_metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"val/{k}", v, epoch)
        self.writer.add_scalar("lr", lr, epoch)

        # Console
        val_dice = val_metrics.get("mean_dice", 0)
        train_loss = train_metrics.get("total", 0)
        val_loss = val_metrics.get("total", 0)
        logger.info(
            f"[Epoch {epoch+1}] "
            f"LR={lr:.6f} "
            f"Train Loss={train_loss:.4f} "
            f"Val Loss={val_loss:.4f} "
            f"Val Dice={val_dice:.4f} "
            f"Best Dice={self.best_dice:.4f}"
        )

        # W&B
        if self.use_wandb and self._wandb:
            self._wandb.log({
                "epoch": epoch + 1,
                "lr": lr,
                **{f"train/{k}": v for k, v in train_metrics.items() if isinstance(v, (int, float))},
                **{f"val/{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))},
            })

    def _save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint."""
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_dice": self.best_dice,
            "metrics": metrics,
            "config": self.config,
        }
        if is_best:
            path = self.output_dir / f"{self.experiment_name}_best.pth"
        else:
            path = self.output_dir / f"{self.experiment_name}_epoch_{epoch+1}.pth"

        torch.save(state, path)
        logger.info(f"Saved checkpoint: {path}")

    def _load_checkpoint(self, path: str):
        """Load checkpoint and restore training state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.epoch = checkpoint.get("epoch", 0) + 1
        self.global_step = checkpoint.get("global_step", 0)
        self.best_dice = checkpoint.get("best_dice", 0.0)
        logger.info(f"Resumed from epoch {self.epoch}, best dice={self.best_dice:.4f}")

    def _save_history(self):
        path = self.output_dir / f"{self.experiment_name}_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2, default=str)
        logger.info(f"Saved training history: {path}")
