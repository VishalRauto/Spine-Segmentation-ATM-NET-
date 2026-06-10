"""
Segmentation and classification metrics for ATM-Net++ evaluation.

Metrics:
- Dice Score (per-class and mean)
- Jaccard / IoU
- HD95 (Hausdorff Distance 95th percentile)
- ASD (Average Surface Distance)
- Precision, Recall, F1
- Classification: Accuracy, AUC, F1
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

logger = logging.getLogger(__name__)


def compute_dice(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    include_background: bool = False,
    smooth: float = 1e-5,
) -> Dict[str, float]:
    """
    Compute per-class and mean Dice scores.

    Args:
        pred: (H, W) or (B, H, W) integer prediction
        target: (H, W) or (B, H, W) integer ground truth
        num_classes: Total number of classes
        include_background: Include class 0

    Returns:
        dict with 'mean_dice', 'class_dice_{c}' for each class
    """
    pred = pred.flatten()
    target = target.flatten()

    start_class = 0 if include_background else 1
    dice_scores = {}
    valid_classes = []

    for c in range(start_class, num_classes):
        p = (pred == c).astype(np.float32)
        t = (target == c).astype(np.float32)

        if t.sum() == 0 and p.sum() == 0:
            continue  # Skip empty classes

        intersection = (p * t).sum()
        denom = p.sum() + t.sum()
        dice = (2 * intersection + smooth) / (denom + smooth)
        dice_scores[f"class_dice_{c}"] = float(dice)
        valid_classes.append(float(dice))

    dice_scores["mean_dice"] = float(np.mean(valid_classes)) if valid_classes else 0.0
    return dice_scores


def compute_iou(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    include_background: bool = False,
    smooth: float = 1e-5,
) -> Dict[str, float]:
    """Compute per-class and mean IoU (Jaccard)."""
    pred = pred.flatten()
    target = target.flatten()
    start_class = 0 if include_background else 1
    iou_scores = {}
    valid = []

    for c in range(start_class, num_classes):
        p = (pred == c).astype(np.float32)
        t = (target == c).astype(np.float32)
        if t.sum() == 0 and p.sum() == 0:
            continue
        intersection = (p * t).sum()
        union = p.sum() + t.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        iou_scores[f"class_iou_{c}"] = float(iou)
        valid.append(float(iou))

    iou_scores["mean_iou"] = float(np.mean(valid)) if valid else 0.0
    return iou_scores


def compute_surface_distances(
    pred_binary: np.ndarray,
    target_binary: np.ndarray,
    spacing: Tuple[float, float] = (1.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute surface distances between prediction and target boundaries.

    Returns:
        (dist_pred_to_gt, dist_gt_to_pred) distances in mm
    """
    # Distance transforms
    dt_pred = distance_transform_edt(~pred_binary.astype(bool), sampling=spacing)
    dt_target = distance_transform_edt(~target_binary.astype(bool), sampling=spacing)

    # Surface voxels
    pred_surface = pred_binary.astype(bool) & (dt_target > 0)
    target_surface = target_binary.astype(bool) & (dt_pred > 0)

    # Distances at surfaces
    if pred_surface.any():
        dist_p2t = dt_target[pred_surface]
    else:
        dist_p2t = np.array([0.0])

    if target_surface.any():
        dist_t2p = dt_pred[target_surface]
    else:
        dist_t2p = np.array([0.0])

    return dist_p2t, dist_t2p


def compute_hd95(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    spacing: Tuple[float, float] = (1.0, 1.0),
    include_background: bool = False,
) -> Dict[str, float]:
    """
    Compute 95th percentile Hausdorff Distance (HD95) per class.

    Args:
        pred: (H, W) integer prediction
        target: (H, W) integer ground truth
        spacing: Physical voxel spacing in mm

    Returns:
        dict with 'mean_hd95' and per-class values
    """
    start_class = 0 if include_background else 1
    hd_scores = {}
    valid = []

    for c in range(start_class, num_classes):
        pred_c = (pred == c)
        target_c = (target == c)

        if not target_c.any() and not pred_c.any():
            continue
        if not target_c.any() or not pred_c.any():
            hd_scores[f"class_hd95_{c}"] = float("inf")
            continue

        try:
            d_p2t, d_t2p = compute_surface_distances(pred_c, target_c, spacing)
            all_distances = np.concatenate([d_p2t, d_t2p])
            hd95 = float(np.percentile(all_distances, 95))
        except Exception:
            hd95 = float("inf")

        hd_scores[f"class_hd95_{c}"] = hd95
        if hd95 != float("inf"):
            valid.append(hd95)

    hd_scores["mean_hd95"] = float(np.mean(valid)) if valid else float("inf")
    return hd_scores


def compute_asd(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    spacing: Tuple[float, float] = (1.0, 1.0),
    include_background: bool = False,
) -> Dict[str, float]:
    """Compute Average Surface Distance (ASD) per class."""
    start_class = 0 if include_background else 1
    asd_scores = {}
    valid = []

    for c in range(start_class, num_classes):
        pred_c = (pred == c)
        target_c = (target == c)

        if not target_c.any() and not pred_c.any():
            continue
        if not target_c.any() or not pred_c.any():
            asd_scores[f"class_asd_{c}"] = float("inf")
            continue

        try:
            d_p2t, d_t2p = compute_surface_distances(pred_c, target_c, spacing)
            asd = float((d_p2t.mean() + d_t2p.mean()) / 2)
        except Exception:
            asd = float("inf")

        asd_scores[f"class_asd_{c}"] = asd
        if asd != float("inf"):
            valid.append(asd)

    asd_scores["mean_asd"] = float(np.mean(valid)) if valid else float("inf")
    return asd_scores


def compute_precision_recall_f1(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    include_background: bool = False,
    smooth: float = 1e-5,
) -> Dict[str, float]:
    """Compute precision, recall, F1 per class and macro average."""
    pred = pred.flatten()
    target = target.flatten()
    start_class = 0 if include_background else 1

    metrics = {}
    precisions, recalls, f1s = [], [], []

    for c in range(start_class, num_classes):
        p = (pred == c).astype(np.float32)
        t = (target == c).astype(np.float32)

        if t.sum() == 0 and p.sum() == 0:
            continue

        tp = (p * t).sum()
        fp = (p * (1 - t)).sum()
        fn = ((1 - p) * t).sum()

        prec = (tp + smooth) / (tp + fp + smooth)
        rec = (tp + smooth) / (tp + fn + smooth)
        f1 = 2 * prec * rec / (prec + rec + smooth)

        metrics[f"precision_{c}"] = float(prec)
        metrics[f"recall_{c}"] = float(rec)
        metrics[f"f1_{c}"] = float(f1)
        precisions.append(float(prec))
        recalls.append(float(rec))
        f1s.append(float(f1))

    metrics["macro_precision"] = float(np.mean(precisions)) if precisions else 0.0
    metrics["macro_recall"] = float(np.mean(recalls)) if recalls else 0.0
    metrics["macro_f1"] = float(np.mean(f1s)) if f1s else 0.0

    return metrics


class SegmentationMetricTracker:
    """
    Tracks and aggregates segmentation metrics over an epoch.
    Accumulates per-batch results and computes final statistics.
    """

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.reset()

    def reset(self):
        self._dice_scores: List[Dict] = []
        self._iou_scores: List[Dict] = []
        self._hd95_scores: List[Dict] = []
        self._prf_scores: List[Dict] = []

    @torch.no_grad()
    def update(
        self,
        pred_logits: torch.Tensor,
        target: torch.Tensor,
        spacing: Tuple[float, float] = (1.0, 1.0),
        compute_hd: bool = False,
    ):
        """
        Update metrics for a batch.

        Args:
            pred_logits: (B, C, H, W) raw logits
            target: (B, H, W) integer ground truth
            spacing: Physical spacing for HD/ASD
            compute_hd: Whether to compute expensive HD95 (slow for large batches)
        """
        pred = pred_logits.argmax(dim=1).cpu().numpy()
        target_np = target.cpu().numpy()

        for b in range(pred.shape[0]):
            self._dice_scores.append(
                compute_dice(pred[b], target_np[b], self.num_classes)
            )
            self._iou_scores.append(
                compute_iou(pred[b], target_np[b], self.num_classes)
            )
            self._prf_scores.append(
                compute_precision_recall_f1(pred[b], target_np[b], self.num_classes)
            )
            if compute_hd:
                self._hd95_scores.append(
                    compute_hd95(pred[b], target_np[b], self.num_classes, spacing)
                )

    def compute(self) -> Dict[str, float]:
        """Compute aggregate metrics over all updates."""
        if not self._dice_scores:
            return {}

        result = {}

        # Mean Dice
        mean_dices = [d["mean_dice"] for d in self._dice_scores]
        result["mean_dice"] = float(np.mean(mean_dices))
        result["std_dice"] = float(np.std(mean_dices))

        # Per-class dice
        for c in range(1, self.num_classes):
            key = f"class_dice_{c}"
            vals = [d[key] for d in self._dice_scores if key in d]
            if vals:
                name = self.class_names[c] if c < len(self.class_names) else f"class_{c}"
                result[f"dice_{name}"] = float(np.mean(vals))

        # Mean IoU
        mean_ious = [d["mean_iou"] for d in self._iou_scores]
        result["mean_iou"] = float(np.mean(mean_ious))

        # Macro F1
        macro_f1s = [d["macro_f1"] for d in self._prf_scores]
        result["macro_f1"] = float(np.mean(macro_f1s))
        result["macro_precision"] = float(np.mean([d["macro_precision"] for d in self._prf_scores]))
        result["macro_recall"] = float(np.mean([d["macro_recall"] for d in self._prf_scores]))

        # HD95
        if self._hd95_scores:
            hd95_vals = [d["mean_hd95"] for d in self._hd95_scores
                        if d.get("mean_hd95", float("inf")) != float("inf")]
            result["mean_hd95"] = float(np.mean(hd95_vals)) if hd95_vals else float("inf")

        return result
