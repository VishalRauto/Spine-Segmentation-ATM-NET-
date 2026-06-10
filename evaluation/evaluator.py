"""
Comprehensive Evaluator for ATM-Net++.
Produces full test-set metrics, per-class analysis, and result exports.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class Evaluator:
    """
    ATM-Net++ evaluation engine.
    Computes: Dice, IoU, HD95, ASD, Precision, Recall, F1
    Also evaluates disease classification accuracy and AUC.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: dict,
        output_dir: Optional[str] = None,
    ):
        self.model = model
        self.device = device
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/evaluation")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        num_classes = config.get("segmentation", {}).get("num_classes", 20)
        class_names = config.get("segmentation", {}).get("class_names", None)

        from training.metrics.segmentation_metrics import SegmentationMetricTracker
        self.metric_tracker = SegmentationMetricTracker(num_classes, class_names)

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        save_results: bool = True,
        compute_hd: bool = True,
        tta: bool = False,
    ) -> Dict[str, float]:
        """
        Full evaluation run.

        Args:
            loader: Test DataLoader
            save_results: Save metrics JSON to disk
            compute_hd: Compute Hausdorff distance (slower)
            tta: Apply test-time augmentation

        Returns:
            Aggregated metrics dict
        """
        self.model.eval()
        self.metric_tracker.reset()

        disease_preds, disease_targets = [], []
        severity_preds, severity_targets = [], []

        for batch in loader:
            batch = {
                k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            if tta:
                seg_logits = self._predict_tta(batch)
            else:
                output = self._forward(batch)
                seg_logits = output["seg_logits"]

            self.metric_tracker.update(
                seg_logits,
                batch["mask"],
                compute_hd=compute_hd,
            )

            # Classification
            if not tta:
                if "disease_label" in batch:
                    disease_preds.append(output["disease"]["pred"].cpu())
                    disease_targets.append(batch["disease_label"].cpu())
                if "severity_label" in batch:
                    severity_preds.append(output["severity"]["pred"].cpu())
                    severity_targets.append(batch["severity_label"].cpu())

        # Aggregated seg metrics
        metrics = self.metric_tracker.compute()

        # Classification metrics
        if disease_preds:
            all_preds = torch.cat(disease_preds).numpy()
            all_targets = torch.cat(disease_targets).numpy()
            cls_metrics = self._classification_metrics(all_preds, all_targets, prefix="disease")
            metrics.update(cls_metrics)

        if severity_preds:
            all_sev_preds = torch.cat(severity_preds).numpy()
            all_sev_targets = torch.cat(severity_targets).numpy()
            sev_metrics = self._classification_metrics(all_sev_preds, all_sev_targets, prefix="severity")
            metrics.update(sev_metrics)

        logger.info("=" * 50)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 50)
        for k, v in sorted(metrics.items()):
            if isinstance(v, float):
                logger.info(f"  {k:40s}: {v:.4f}")
        logger.info("=" * 50)

        if save_results:
            out_path = self.output_dir / "test_metrics.json"
            with open(out_path, "w") as f:
                json.dump({k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}, f, indent=2)
            logger.info(f"Saved metrics to {out_path}")

        return metrics

    def _forward(self, batch: Dict) -> Dict:
        """Run model forward pass."""
        inputs = {"image": batch["image"]}
        if "input_ids" in batch:
            inputs["input_ids"] = batch["input_ids"]
            inputs["attention_mask"] = batch["attention_mask"]
        if "demographics" in batch:
            inputs["demographics"] = batch["demographics"]
        return self.model(**inputs)

    def _predict_tta(self, batch: Dict) -> torch.Tensor:
        """Test-time augmentation: average predictions over flipped versions."""
        from datasets.transforms.augmentations import TestTimeAugmentation

        image = batch["image"]
        preds = []

        for transform in TestTimeAugmentation.TRANSFORMS:
            aug_image = torch.from_numpy(
                TestTimeAugmentation.apply(image.cpu().numpy(), transform)
            ).to(self.device)

            aug_batch = dict(batch)
            aug_batch["image"] = aug_image
            output = self._forward(aug_batch)
            seg_probs = F.softmax(output["seg_logits"], dim=1).cpu().numpy()
            seg_probs_rev = TestTimeAugmentation.reverse(seg_probs, transform)
            preds.append(torch.from_numpy(seg_probs_rev))

        # Average predictions
        avg_pred = torch.stack(preds).mean(dim=0).to(self.device)
        return avg_pred  # Already probabilities, not logits

    def _classification_metrics(
        self,
        preds: np.ndarray,
        targets: np.ndarray,
        prefix: str,
    ) -> Dict[str, float]:
        """Compute classification accuracy and macro F1."""
        from sklearn.metrics import accuracy_score, f1_score

        acc = float(accuracy_score(targets, preds))
        f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
        return {
            f"{prefix}_accuracy": acc,
            f"{prefix}_macro_f1": f1,
        }

    def export_predictions(
        self,
        loader: DataLoader,
        output_dir: str,
        save_masks: bool = True,
    ):
        """Export segmentation predictions to disk for visual inspection."""
        import cv2
        from datasets.preprocessing.label_mapper import create_colorized_mask

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.eval()

        for i, batch in enumerate(loader):
            batch = {
                k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            output = self._forward(batch)
            pred_mask = output["seg_logits"].argmax(dim=1).cpu().numpy()
            image = batch["image"].cpu().numpy()
            gt_mask = batch["mask"].cpu().numpy()

            for b in range(pred_mask.shape[0]):
                pid = batch.get("patient_id", [f"sample_{i}_{b}"])[b]
                mod = batch.get("modality", ["T2"])[b]
                s_idx = batch.get("slice_idx", [0])[b]
                fname = f"{pid}_{mod}_slice{s_idx:03d}"

                # Save colorized prediction
                pred_rgb = create_colorized_mask(pred_mask[b])
                cv2.imwrite(str(out_dir / f"{fname}_pred.png"),
                            cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))

                # Save GT
                gt_rgb = create_colorized_mask(gt_mask[b])
                cv2.imwrite(str(out_dir / f"{fname}_gt.png"),
                            cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR))

                # Save raw image
                img = (np.clip(image[b, 0], 0, 1) * 255).astype(np.uint8)
                cv2.imwrite(str(out_dir / f"{fname}_image.png"), img)

                if i == 0 and b == 0:
                    logger.info(f"Sample export: {out_dir / fname}_*.png")

            if i >= 49:  # Limit exports
                break

        logger.info(f"Exported predictions to {out_dir}")
