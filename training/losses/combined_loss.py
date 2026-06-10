"""
Combined loss functions for ATM-Net++ multi-task training.

Implements:
- Dice Loss (segmentation)
- Focal Loss (segmentation + classification)
- Boundary Loss (segmentation)
- Cross-entropy (classification)
- Contrastive alignment loss (image-text)
- Deep supervision loss wrapper
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Dice Loss
# ─────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Soft Dice Loss for multi-class segmentation.
    Supports per-class weighting and background exclusion.
    """

    def __init__(
        self,
        num_classes: int,
        smooth: float = 1e-5,
        include_background: bool = False,
        class_weights: Optional[torch.Tensor] = None,
        softmax: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.include_background = include_background
        self.softmax = softmax
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) raw logits
            target: (B, H, W) integer class labels

        Returns:
            Scalar Dice loss
        """
        if self.softmax:
            pred_soft = F.softmax(pred, dim=1)
        else:
            pred_soft = pred

        B, C, H, W = pred_soft.shape

        # One-hot encode target: (B, C, H, W)
        target_one_hot = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float()

        # Flatten spatial dims: (B, C, H*W)
        p = pred_soft.view(B, C, -1)
        t = target_one_hot.view(B, C, -1)

        start_class = 0 if self.include_background else 1
        dice_per_class = []

        for c in range(start_class, C):
            pc = p[:, c, :]
            tc = t[:, c, :]
            intersection = (pc * tc).sum(dim=-1)
            union = pc.sum(dim=-1) + tc.sum(dim=-1)
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_per_class.append(dice.mean())

        dice_per_class = torch.stack(dice_per_class)

        if self.class_weights is not None:
            w = self.class_weights[start_class:].to(pred.device)
            if w.shape[0] == dice_per_class.shape[0]:
                dice_per_class = dice_per_class * w / w.sum()

        return 1.0 - dice_per_class.mean()


# ─────────────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss (Lin et al., 2017).
    Addresses class imbalance by down-weighting easy examples.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        ignore_index: int = -1,
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) or (B, C) logits
            target: (B, H, W) or (B,) integer labels
        """
        is_seg = pred.dim() == 4
        B, C = pred.shape[:2]

        if is_seg:
            pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, C)
            target_flat = target.reshape(-1)
        else:
            pred_flat = pred
            target_flat = target

        # Filter ignore index
        if self.ignore_index >= 0:
            valid = target_flat != self.ignore_index
            pred_flat = pred_flat[valid]
            target_flat = target_flat[valid]

        log_prob = F.log_softmax(pred_flat, dim=-1)
        prob = torch.exp(log_prob)

        # Gather probabilities of the target class
        target_log_prob = log_prob.gather(-1, target_flat.unsqueeze(-1)).squeeze(-1)
        target_prob = prob.gather(-1, target_flat.unsqueeze(-1)).squeeze(-1)

        # Focal weight
        focal_weight = (1 - target_prob) ** self.gamma

        # Class weight
        if self.alpha is not None:
            alpha_t = self.alpha.to(pred.device)[target_flat]
            focal_weight = focal_weight * alpha_t

        loss = -focal_weight * target_log_prob

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ─────────────────────────────────────────────────────────────────────
# Boundary Loss
# ─────────────────────────────────────────────────────────────────────

class BoundaryLoss(nn.Module):
    """
    Boundary Loss (Kervadec et al., 2019).
    Uses distance transform of ground truth boundaries.
    Penalizes predictions far from boundaries.
    """

    def __init__(self, num_classes: int, include_background: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.include_background = include_background

    def _compute_dist_map(self, seg_gt: np.ndarray) -> np.ndarray:
        """Compute signed distance transform for each class."""
        dist_maps = np.zeros_like(seg_gt, dtype=np.float32)
        for b in range(seg_gt.shape[0]):
            for c in range(1 if not self.include_background else 0, self.num_classes):
                binary = (seg_gt[b] == c).astype(np.uint8)
                if binary.sum() == 0:
                    continue
                # Distance to boundary
                dt_in = distance_transform_edt(binary)
                dt_out = distance_transform_edt(1 - binary)
                dist_maps[b] += (dt_out - dt_in) * (c == seg_gt[b])
        return dist_maps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) softmax probabilities
            target: (B, H, W) integer labels
        """
        B, C, H, W = pred.shape
        target_np = target.cpu().numpy()

        # Build distance maps (expensive but high-quality boundary signal)
        dist_maps = self._compute_dist_map(target_np)
        dist_tensor = torch.from_numpy(dist_maps).float().to(pred.device)

        # Multiply softmax output by distance map
        pred_soft = F.softmax(pred, dim=1)

        # For each class, compute weighted prediction
        start = 0 if self.include_background else 1
        loss = torch.tensor(0.0, device=pred.device)
        n = 0
        for c in range(start, C):
            d = dist_tensor  # Using combined dist map
            loss = loss + (pred_soft[:, c, :, :] * d).mean()
            n += 1

        return loss / max(n, 1)


# ─────────────────────────────────────────────────────────────────────
# Contrastive Alignment Loss
# ─────────────────────────────────────────────────────────────────────

class ContrastiveAlignmentLoss(nn.Module):
    """
    NT-Xent style contrastive loss for image-text alignment.
    Ensures image and text features are aligned in embedding space.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        image_proj: torch.Tensor,
        text_proj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image_proj: (B, D) L2-normalized image embeddings
            text_proj: (B, D) L2-normalized text embeddings

        Returns:
            Symmetric contrastive loss
        """
        B = image_proj.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=image_proj.device)

        # Similarity matrix
        sim = torch.mm(image_proj, text_proj.t()) / self.temperature  # (B, B)

        # Labels: diagonal (i, i) are positive pairs
        labels = torch.arange(B, device=image_proj.device)

        loss_i2t = F.cross_entropy(sim, labels)
        loss_t2i = F.cross_entropy(sim.t(), labels)

        return (loss_i2t + loss_t2i) / 2


# ─────────────────────────────────────────────────────────────────────
# Combined ATM-Net++ Loss
# ─────────────────────────────────────────────────────────────────────

class ATMNetLoss(nn.Module):
    """
    Combined multi-task loss for ATM-Net++.

    Tasks:
    1. Segmentation: Dice + Focal + Boundary (weighted sum)
    2. Disease classification: Focal CE
    3. Severity estimation: CE + MSE regression
    4. Level localization: Binary CE
    5. IVD pathology: Multi-label BCE + Pfirrmann regression
    6. Report generation: auxiliary CE
    7. Feature alignment: Contrastive
    8. Deep supervision: Dice on intermediate outputs
    """

    def __init__(
        self,
        num_seg_classes: int = 20,
        num_disease_classes: int = 7,
        seg_dice_weight: float = 1.0,
        seg_focal_weight: float = 0.5,
        seg_boundary_weight: float = 0.2,
        cls_weight: float = 0.3,
        severity_weight: float = 0.2,
        level_weight: float = 0.2,
        report_weight: float = 0.1,
        contrastive_weight: float = 0.05,
        ds_weight: float = 0.4,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.seg_dice_weight = seg_dice_weight
        self.seg_focal_weight = seg_focal_weight
        self.seg_boundary_weight = seg_boundary_weight
        self.cls_weight = cls_weight
        self.severity_weight = severity_weight
        self.level_weight = level_weight
        self.report_weight = report_weight
        self.contrastive_weight = contrastive_weight
        self.ds_weight = ds_weight

        # Segmentation losses
        self.dice_loss = DiceLoss(
            num_seg_classes,
            include_background=False,
            class_weights=class_weights,
        )
        self.focal_loss = FocalLoss(gamma=2.0)
        self.boundary_loss = BoundaryLoss(num_seg_classes)

        # Classification losses
        if class_weights is not None:
            cls_w = class_weights[:num_disease_classes] if len(class_weights) >= num_disease_classes else None
        else:
            cls_w = None
        self.disease_focal = FocalLoss(gamma=2.0, alpha=cls_w)
        self.severity_ce = nn.CrossEntropyLoss()
        self.level_bce = nn.BCEWithLogitsLoss()
        self.ivd_bce = nn.BCEWithLogitsLoss()
        self.pfirrmann_mse = nn.MSELoss()
        self.contrastive = ContrastiveAlignmentLoss(temperature=0.07)

    def forward(
        self,
        output: Dict,
        batch: Dict,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.

        Args:
            output: Model output dict (from ATMNetPlusPlus.forward)
            batch: Data batch dict (from DataLoader)

        Returns:
            dict with 'total', 'seg', 'cls', 'severity', 'level', 'contrastive', etc.
        """
        losses = {}
        device = output["seg_logits"].device

        # ── Segmentation loss ────────────────────────────────────────
        seg_target = batch["mask"].to(device)

        dice = self.dice_loss(output["seg_logits"], seg_target)
        focal = self.focal_loss(output["seg_logits"], seg_target)
        losses["seg_dice"] = dice
        losses["seg_focal"] = focal

        # Boundary loss (expensive, optional)
        try:
            boundary = self.boundary_loss(
                F.softmax(output["seg_logits"], dim=1), seg_target
            )
            losses["seg_boundary"] = boundary
        except Exception:
            losses["seg_boundary"] = torch.tensor(0.0, device=device)

        losses["seg"] = (
            self.seg_dice_weight * dice
            + self.seg_focal_weight * focal
            + self.seg_boundary_weight * losses["seg_boundary"]
        )

        # ── Deep supervision ─────────────────────────────────────────
        if "ds_logits" in output and output["ds_logits"]:
            ds_loss = torch.tensor(0.0, device=device)
            for ds_logit in output["ds_logits"]:
                ds_loss = ds_loss + self.dice_loss(ds_logit, seg_target)
            losses["ds"] = ds_loss / len(output["ds_logits"])
            losses["seg"] = losses["seg"] + self.ds_weight * losses["ds"]

        # ── Disease classification ────────────────────────────────────
        if "disease_label" in batch:
            dis_target = batch["disease_label"].to(device)
            dis_logits = output["disease"]["logits"]
            losses["cls"] = self.disease_focal(dis_logits, dis_target)
        else:
            losses["cls"] = torch.tensor(0.0, device=device)

        # ── Severity ─────────────────────────────────────────────────
        if "severity_label" in batch:
            sev_target = batch["severity_label"].to(device)
            sev_logits = output["severity"]["logits"]
            losses["severity"] = self.severity_ce(sev_logits, sev_target)
        else:
            losses["severity"] = torch.tensor(0.0, device=device)

        # ── Level localization ────────────────────────────────────────
        # We approximate level labels from segmentation (presence of disc classes)
        if "level" in output:
            # Auto-derive level targets from seg mask
            level_targets = self._derive_level_targets(seg_target)
            lvl_logits = output["level"]["logits"]
            losses["level"] = self.level_bce(lvl_logits, level_targets)
        else:
            losses["level"] = torch.tensor(0.0, device=device)

        # ── IVD pathology multi-label ─────────────────────────────────
        if "ivd_pathology" in output and all(
            k in batch for k in ["disc_herniation", "disc_bulging", "disc_narrowing"]
        ):
            ivd_targets = torch.stack([
                batch.get("disc_herniation", torch.zeros(seg_target.shape[0])),
                batch.get("disc_bulging", torch.zeros(seg_target.shape[0])),
                batch.get("disc_narrowing", torch.zeros(seg_target.shape[0])),
                batch.get("spondylolisthesis", torch.zeros(seg_target.shape[0])),
            ], dim=-1).to(device)

            if output["ivd_pathology"]["logits"].shape[-1] >= 4:
                ivd_logits = output["ivd_pathology"]["logits"][:, :4]
                losses["ivd"] = self.ivd_bce(ivd_logits, ivd_targets)
            else:
                losses["ivd"] = torch.tensor(0.0, device=device)

            # Pfirrmann regression
            if "pfirrmann_grade" in batch:
                pf_pred = output["ivd_pathology"]["pfirrmann_score"]
                pf_target = batch["pfirrmann_grade"].to(device) * 4 + 1  # Scale to [1,5]
                losses["pfirrmann"] = self.pfirrmann_mse(pf_pred, pf_target)
        else:
            losses["ivd"] = torch.tensor(0.0, device=device)
            losses["pfirrmann"] = torch.tensor(0.0, device=device)

        # ── Report auxiliary loss ─────────────────────────────────────
        if "disease_label" in batch and "report_disease_logits" in output:
            losses["report"] = self.disease_focal(
                output["report_disease_logits"], batch["disease_label"].to(device)
            )
        else:
            losses["report"] = torch.tensor(0.0, device=device)

        # ── Contrastive alignment ─────────────────────────────────────
        if "img_proj" in output and "txt_proj" in output:
            losses["contrastive"] = self.contrastive(
                output["img_proj"], output["txt_proj"]
            )
        else:
            losses["contrastive"] = torch.tensor(0.0, device=device)

        # ── Total loss ────────────────────────────────────────────────
        losses["total"] = (
            losses["seg"]
            + self.cls_weight * losses["cls"]
            + self.severity_weight * losses["severity"]
            + self.level_weight * losses["level"]
            + self.report_weight * losses["report"]
            + self.contrastive_weight * losses["contrastive"]
            + 0.1 * losses["ivd"]
            + 0.05 * losses["pfirrmann"]
        )

        return losses

    def _derive_level_targets(self, seg_mask: torch.Tensor) -> torch.Tensor:
        """
        Derive level presence targets from segmentation mask.
        Disc classes 10-17 correspond to levels 0-7.
        """
        B = seg_mask.shape[0]
        # Classes 10-17 = 8 disc levels
        disc_class_start = 10
        num_levels = 8
        targets = torch.zeros(B, num_levels, device=seg_mask.device)
        for i in range(num_levels):
            cls = disc_class_start + i
            targets[:, i] = (seg_mask == cls).float().sum(dim=[1, 2]) > 0
        return targets.float()
