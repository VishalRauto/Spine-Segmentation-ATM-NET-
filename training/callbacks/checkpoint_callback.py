"""
Checkpoint and logging callbacks for ATM-Net++ training.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelCheckpoint:
    """
    Saves model checkpoints based on monitored metric.
    Keeps top-K best checkpoints and the latest one.
    """

    def __init__(
        self,
        save_dir: str,
        monitor: str = "val_dice",
        mode: str = "max",
        save_top_k: int = 3,
        save_last: bool = True,
        filename_prefix: str = "atmnet_pp",
        verbose: bool = True,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.filename_prefix = filename_prefix
        self.verbose = verbose

        self._best_value = float("-inf") if mode == "max" else float("inf")
        self._checkpoints: List[Dict] = []  # sorted list of saved checkpoints

    def _is_better(self, current: float) -> bool:
        if self.mode == "max":
            return current > self._best_value
        return current < self._best_value

    def __call__(
        self,
        epoch: int,
        metrics: Dict[str, float],
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler=None,
        extra: Optional[Dict] = None,
    ) -> bool:
        """
        Check if we should save and save if so.

        Returns:
            True if this was a new best checkpoint.
        """
        value = metrics.get(self.monitor, 0.0)
        is_best = self._is_better(value)

        # Always save latest
        if self.save_last:
            last_path = self.save_dir / f"{self.filename_prefix}_last.pth"
            self._save(last_path, epoch, metrics, model, optimizer, scaler, extra)

        if is_best:
            self._best_value = value
            best_path = self.save_dir / f"{self.filename_prefix}_best.pth"
            self._save(best_path, epoch, metrics, model, optimizer, scaler, extra)
            if self.verbose:
                logger.info(f"[Checkpoint] New best {self.monitor}={value:.4f} → {best_path.name}")

        # Save periodic checkpoint
        epoch_path = self.save_dir / f"{self.filename_prefix}_epoch{epoch:04d}.pth"
        self._save(epoch_path, epoch, metrics, model, optimizer, scaler, extra)
        self._checkpoints.append({"epoch": epoch, "value": value, "path": str(epoch_path)})

        # Prune old checkpoints (keep top-k + last)
        self._prune_checkpoints()

        return is_best

    def _save(
        self,
        path: Path,
        epoch: int,
        metrics: Dict,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler=None,
        extra: Optional[Dict] = None,
    ):
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "best_value": self._best_value,
        }
        if scaler is not None:
            state["scaler_state_dict"] = scaler.state_dict()
        if extra:
            state.update(extra)
        torch.save(state, path)

    def _prune_checkpoints(self):
        if len(self._checkpoints) <= self.save_top_k:
            return

        # Sort by value (descending for max, ascending for min)
        reverse = self.mode == "max"
        sorted_ckpts = sorted(
            self._checkpoints,
            key=lambda x: x["value"],
            reverse=reverse,
        )
        # Keep top-k
        to_keep = {c["path"] for c in sorted_ckpts[:self.save_top_k]}
        to_delete = [c for c in self._checkpoints if c["path"] not in to_keep]

        for ckpt in to_delete:
            p = Path(ckpt["path"])
            if p.exists():
                p.unlink()
                logger.debug(f"[Checkpoint] Removed old checkpoint: {p.name}")

        self._checkpoints = sorted_ckpts[:self.save_top_k]

    @property
    def best_value(self) -> float:
        return self._best_value

    @property
    def best_checkpoint_path(self) -> str:
        return str(self.save_dir / f"{self.filename_prefix}_best.pth")


class TensorBoardImageLogger:
    """
    Logs segmentation predictions as images to TensorBoard during training.
    """

    def __init__(
        self,
        writer,
        log_every_n_epochs: int = 5,
        num_images: int = 4,
    ):
        self.writer = writer
        self.log_every_n_epochs = log_every_n_epochs
        self.num_images = num_images

    def log_segmentation(
        self,
        epoch: int,
        images: torch.Tensor,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
        tag_prefix: str = "val",
    ):
        if epoch % self.log_every_n_epochs != 0:
            return

        import numpy as np
        from datasets.preprocessing.label_mapper import create_colorized_mask, ATMNET_COLORMAP

        n = min(self.num_images, images.shape[0])
        for i in range(n):
            img_np = images[i, 0].cpu().numpy()
            pred_np = pred_masks[i].cpu().numpy()
            gt_np = gt_masks[i].cpu().numpy()

            # Normalize image
            img_norm = np.clip(img_np, 0, 1)

            # Colorize masks
            pred_rgb = create_colorized_mask(pred_np).transpose(2, 0, 1) / 255.0
            gt_rgb = create_colorized_mask(gt_np).transpose(2, 0, 1) / 255.0

            self.writer.add_image(f"{tag_prefix}/image_{i}", img_np[np.newaxis], epoch)
            self.writer.add_image(f"{tag_prefix}/pred_mask_{i}", pred_rgb, epoch)
            self.writer.add_image(f"{tag_prefix}/gt_mask_{i}", gt_rgb, epoch)


class ProgressLogger:
    """
    Rich/tqdm-based progress logging for training.
    """

    def __init__(self, total_epochs: int, use_rich: bool = True):
        self.total_epochs = total_epochs
        self.use_rich = use_rich
        self._progress = None

        if use_rich:
            try:
                from rich.progress import (
                    Progress, SpinnerColumn, BarColumn,
                    TextColumn, TimeRemainingColumn, MofNCompleteColumn
                )
                self._progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeRemainingColumn(),
                    transient=False,
                )
                self._task = self._progress.add_task("Training", total=total_epochs)
                self._progress.start()
            except ImportError:
                self.use_rich = False

    def update(self, epoch: int, metrics: Dict[str, float]):
        desc = (
            f"Epoch {epoch+1}/{self.total_epochs} | "
            f"Dice={metrics.get('mean_dice', 0):.4f} | "
            f"Loss={metrics.get('total', 0):.4f}"
        )
        if self.use_rich and self._progress:
            self._progress.update(self._task, advance=1, description=desc)
        else:
            logger.info(desc)

    def close(self):
        if self.use_rich and self._progress:
            self._progress.stop()
