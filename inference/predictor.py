"""
ATM-Net++ Inference Engine.

Handles:
- Single image inference
- Batch inference
- DICOM / NIfTI / MHA / PNG input
- Test-time augmentation
- Post-processing (morphological cleanup, connected components)
- Result packaging for API responses
"""

from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SpinePredictor:
    """
    Production inference engine for ATM-Net++.
    Thread-safe. Loads model once, runs predictions on demand.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        config: Optional[dict] = None,
        threshold: float = 0.5,
        use_tta: bool = False,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.config = config or {}
        self.threshold = threshold
        self.use_tta = use_tta

        # Build preprocessor
        from datasets.preprocessing.normalizer import SpinePreprocessor, NormalizationStrategy
        data_cfg = self.config.get("data", {})
        self.preprocessor = SpinePreprocessor(
            target_size=tuple(data_cfg.get("image_size", [512, 512])),
            normalize_strategy=NormalizationStrategy.PERCENTILE,
            add_channel_dim=True,
        )

        # Class info
        seg_cfg = self.config.get("segmentation", {})
        self.num_classes = seg_cfg.get("num_classes", 20)
        self.class_names = seg_cfg.get("class_names", [f"class_{i}" for i in range(self.num_classes)])

        from datasets.preprocessing.label_mapper import ATMNET_COLORMAP, ATMNET_TO_NAME
        self.colormap = ATMNET_COLORMAP
        self.class_name_map = ATMNET_TO_NAME

        from models.report_generator.clinical_report import TemplateReportGenerator
        self.report_generator = TemplateReportGenerator()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config: dict,
        device: Optional[torch.device] = None,
    ) -> "SpinePredictor":
        """Load predictor from a saved checkpoint."""
        from models.atmnet_plus_plus import ATMNetPlusPlus

        seg_cfg = config.get("segmentation", {})
        cls_cfg = config.get("classification", {})
        model_cfg = config.get("model", {})
        fusion_cfg = config.get("fusion", {})

        model = ATMNetPlusPlus(
            img_size=tuple(model_cfg.get("img_size", [512, 512])),
            in_channels=model_cfg.get("in_channels", 1),
            num_seg_classes=seg_cfg.get("num_classes", 20),
            num_disease_classes=cls_cfg.get("num_disease_classes", 7),
            feature_size=model_cfg.get("feature_size", 48),
            fusion_dim=fusion_cfg.get("fusion_dim", 512),
        )

        d = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=d)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint.get('epoch', '?')})")

        return cls(model=model, device=d, config=config)

    # ──────────────────────────────────────────────────────────────────
    # Primary prediction entry point
    # ──────────────────────────────────────────────────────────────────

    def predict_from_file(
        self,
        image_path: str,
        report_text: Optional[str] = None,
        demographics: Optional[Dict] = None,
        modality: str = "T2",
    ) -> Dict[str, Any]:
        """
        Complete prediction from a file path.

        Args:
            image_path: Path to MHA/NIfTI/DICOM/PNG file
            report_text: Optional radiology report text
            demographics: Optional patient demographics dict
            modality: "T1" or "T2"

        Returns:
            Complete prediction result dict
        """
        t_start = time.time()

        # Load and preprocess image
        slices = self._load_image_file(image_path, modality)

        # Run prediction on all slices
        all_results = []
        for slice_img in slices:
            result = self.predict_slice(
                image_array=slice_img,
                report_text=report_text,
                demographics=demographics,
                modality=modality,
            )
            all_results.append(result)

        # Aggregate multi-slice results
        final = self._aggregate_results(all_results, slices)
        final["inference_time_ms"] = round((time.time() - t_start) * 1000, 2)
        final["num_slices_processed"] = len(slices)
        final["source_file"] = str(Path(image_path).name)

        return final

    def predict_slice(
        self,
        image_array: np.ndarray,
        report_text: Optional[str] = None,
        demographics: Optional[Dict] = None,
        modality: str = "T2",
    ) -> Dict[str, Any]:
        """
        Predict on a single 2D MRI slice.

        Args:
            image_array: (H, W) float32 or uint8 image
            report_text: Optional radiology report
            demographics: Optional demographics dict
            modality: MRI modality

        Returns:
            Prediction result dict
        """
        # Preprocess
        img_processed = self.preprocessor.process_slice(
            image_array.astype(np.float32), modality=modality
        )  # (1, H, W)

        # Build input tensor
        image_tensor = torch.from_numpy(img_processed).unsqueeze(0).to(self.device)  # (1, 1, H, W)

        # Build text inputs
        text_inputs = self._encode_text(report_text)

        # Build demographic tensor
        demo_tensor = self._encode_demographics(demographics)

        # Forward pass
        with torch.no_grad():
            if self.use_tta:
                seg_probs, cls_output = self._predict_with_tta(image_tensor, text_inputs, demo_tensor)
            else:
                output = self.model(
                    image=image_tensor,
                    demographics=demo_tensor,
                    **text_inputs,
                )
                seg_probs = F.softmax(output["seg_logits"], dim=1)
                cls_output = output

        # Post-process segmentation
        seg_pred = seg_probs.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)
        seg_probs_np = seg_probs.squeeze(0).cpu().numpy()            # (num_classes, H, W)
        seg_pred_clean = self._postprocess_segmentation(seg_pred)

        # Extract classification results
        disease_pred = int(cls_output["disease"]["pred"][0].item())
        disease_conf = float(cls_output["disease"]["confidence"][0].item())
        disease_probs = cls_output["disease"]["probs"][0].cpu().numpy().tolist()
        severity_pred = int(cls_output["severity"]["pred"][0].item())
        level_pred = cls_output["level"]["pred"][0].cpu().numpy().tolist()
        pfirrmann = float(cls_output["ivd_pathology"]["pfirrmann_score"][0].item())

        # Generate report
        pred_dict = {
            "disease_pred": disease_pred,
            "disease_confidence": disease_conf,
            "severity_pred": severity_pred,
            "level_pred": level_pred,
            "pfirrmann_score": pfirrmann,
        }
        report = self.report_generator.generate(
            pred_dict,
            patient_info=demographics,
        )

        # Grad-CAM
        gradcam_b64 = self._compute_gradcam(image_tensor, cls_output, disease_pred, img_processed[0])

        # Segmentation overlay
        seg_overlay_b64 = self._create_seg_overlay(img_processed[0], seg_pred_clean)

        # Class distribution
        class_distribution = self._compute_class_distribution(seg_pred_clean)

        return {
            "segmentation": {
                "mask": seg_pred_clean.tolist(),
                "overlay_b64": seg_overlay_b64,
                "class_distribution": class_distribution,
                "detected_structures": self._identify_detected_structures(seg_pred_clean),
            },
            "classification": {
                "disease_id": disease_pred,
                "disease_name": self.class_name_map.get(disease_pred, f"Unknown({disease_pred})"),
                "confidence": round(disease_conf, 4),
                "disease_probabilities": {
                    name: round(p, 4)
                    for name, p in zip(
                        self.config.get("classification", {}).get("disease_names",
                            ["Normal","Disc_Herniation","Disc_Bulge","Spinal_Stenosis","DDD","Spondylolisthesis","Fracture"]),
                        disease_probs,
                    )
                },
            },
            "severity": {
                "id": severity_pred,
                "name": ["Mild", "Moderate", "Severe"][severity_pred],
            },
            "levels": {
                "affected": [
                    name for name, active in zip(
                        ["T10/T11","T11/T12","T12/L1","L1/L2","L2/L3","L3/L4","L4/L5","L5/S1"],
                        level_pred
                    ) if active > 0.5
                ],
                "all_probs": {
                    name: round(float(p), 4)
                    for name, p in zip(
                        ["T10/T11","T11/T12","T12/L1","L1/L2","L2/L3","L3/L4","L4/L5","L5/S1"],
                        level_pred,
                    )
                },
            },
            "pfirrmann_grade": round(pfirrmann, 2),
            "report": report,
            "gradcam_b64": gradcam_b64,
        }

    # ──────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────

    def _load_image_file(self, path: str, modality: str) -> List[np.ndarray]:
        """Load image file and return list of 2D slices."""
        ext = "".join(Path(path).suffixes).lower()

        if ext in {".mha", ".mhd", ".nii", ".nii.gz"}:
            from datasets.preprocessing.mha_reader import MedicalImageReader
            reader = MedicalImageReader()
            med_img = reader.read(path)
            data = med_img.data.astype(np.float32)
            # Return all slices along first axis
            if data.ndim == 3:
                return [data[i] for i in range(data.shape[0])]
            return [data]
        elif ext in {".dcm"}:
            from datasets.preprocessing.mha_reader import MedicalImageReader
            reader = MedicalImageReader()
            med_img = reader._read_dicom(Path(path))
            data = med_img.data.astype(np.float32)
            if data.ndim == 3:
                return [data[i] for i in range(data.shape[0])]
            return [data]
        elif ext in {".png", ".jpg", ".jpeg"}:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Cannot read image: {path}")
            return [img.astype(np.float32)]
        else:
            # Try SimpleITK as fallback
            from datasets.preprocessing.mha_reader import MedicalImageReader
            reader = MedicalImageReader()
            med_img = reader.read(path)
            data = med_img.data.astype(np.float32)
            if data.ndim == 3:
                return [data[i] for i in range(data.shape[0])]
            return [data]

    def _encode_text(self, report_text: Optional[str]) -> Dict:
        """Tokenize report text for model input."""
        if report_text is None or not hasattr(self.model, "text_encoder") or self.model.text_encoder is None:
            return {}
        tokenizer = self.model.text_encoder.tokenizer
        if tokenizer is None:
            return {}
        try:
            enc = self.model.text_encoder.tokenize([report_text], device=self.device)
            return enc
        except Exception:
            return {}

    def _encode_demographics(self, demographics: Optional[Dict]) -> Optional[torch.Tensor]:
        """Convert demographics dict to tensor."""
        if demographics is None:
            return torch.zeros(1, 8, device=self.device)
        # Map fields to 8-dim vector
        feat = np.zeros(8, dtype=np.float32)
        sex = str(demographics.get("sex", demographics.get("gender", "F"))).strip().upper()
        feat[0] = 1.0 if sex.startswith("M") else 0.0
        age = demographics.get("age", 50)
        feat[1] = np.clip(float(age) / 80.0, 0, 1)
        bmi = demographics.get("bmi", 25)
        feat[2] = np.clip(float(bmi) / 40.0, 0, 1)
        height = demographics.get("height", 170)
        feat[3] = np.clip(float(height) / 200.0, 0, 1)
        weight = demographics.get("weight", 70)
        feat[4] = np.clip(float(weight) / 150.0, 0, 1)
        feat[5] = 0.5  # field strength default
        feat[6] = 0.5  # echo time default
        feat[7] = 0.5  # rep time default
        return torch.from_numpy(feat).unsqueeze(0).to(self.device)

    def _predict_with_tta(
        self,
        image_tensor: torch.Tensor,
        text_inputs: Dict,
        demo_tensor: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict]:
        """Run TTA inference."""
        from datasets.transforms.augmentations import TestTimeAugmentation

        all_probs = []
        last_output = None

        for transform in TestTimeAugmentation.TRANSFORMS:
            aug_img_np = TestTimeAugmentation.apply(image_tensor.cpu().numpy(), transform)
            aug_img = torch.from_numpy(aug_img_np).to(self.device)

            output = self.model(image=aug_img, demographics=demo_tensor, **text_inputs)
            probs = F.softmax(output["seg_logits"], dim=1).cpu().numpy()
            probs_rev = TestTimeAugmentation.reverse(probs, transform)
            all_probs.append(torch.from_numpy(probs_rev))
            last_output = output

        avg_probs = torch.stack(all_probs).mean(dim=0).to(self.device)
        return avg_probs, last_output

    def _postprocess_segmentation(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological post-processing to clean up segmentation.
        - Remove small isolated regions
        - Fill small holes
        """
        from scipy import ndimage
        cleaned = mask.copy()
        for c in range(1, self.num_classes):
            binary = (cleaned == c).astype(np.uint8)
            if binary.sum() < 50:  # Ignore tiny regions
                cleaned[cleaned == c] = 0
                continue
            # Fill small holes
            filled = ndimage.binary_fill_holes(binary).astype(np.uint8)
            # Keep only largest connected component
            labeled, n_components = ndimage.label(filled)
            if n_components > 1:
                sizes = ndimage.sum(binary, labeled, range(1, n_components + 1))
                largest = np.argmax(sizes) + 1
                filled = (labeled == largest).astype(np.uint8)
            # Write back
            cleaned[cleaned == c] = 0
            cleaned[filled == 1] = c
        return cleaned

    def _compute_gradcam(
        self,
        image_tensor: torch.Tensor,
        model_output: Dict,
        target_class: int,
        raw_image: np.ndarray,
    ) -> str:
        """Compute Grad-CAM and return base64 encoded overlay."""
        try:
            from models.explainability.grad_cam import GradCAM, ExplainabilityVisualizer

            # Find target layer
            target_layer = None
            for name, module in self.model.named_modules():
                if hasattr(module, "conv2") and isinstance(module.conv2, torch.nn.Conv2d):
                    target_layer = module.conv2
                    break
            if target_layer is None:
                # Fallback to any conv layer
                for name, module in self.model.named_modules():
                    if isinstance(module, torch.nn.Conv2d) and module.out_channels >= 32:
                        target_layer = module
                        break

            if target_layer is None:
                return ""

            # Need grad so re-run
            img_grad = image_tensor.clone().requires_grad_(True)
            gradcam = GradCAM(self.model, target_layer)
            cam = gradcam.generate(img_grad, target_class=target_class)
            gradcam.remove_hooks()

            overlay = ExplainabilityVisualizer.create_heatmap_overlay(raw_image, cam)
            b64 = ExplainabilityVisualizer.encode_image_b64(overlay)
            return b64
        except Exception as e:
            logger.debug(f"GradCAM failed: {e}")
            return ""

    def _create_seg_overlay(self, image: np.ndarray, mask: np.ndarray) -> str:
        """Create segmentation overlay and return as base64."""
        try:
            from models.explainability.grad_cam import ExplainabilityVisualizer
            overlay = ExplainabilityVisualizer.create_segmentation_overlay(
                image, mask, self.colormap, alpha=0.5
            )
            return ExplainabilityVisualizer.encode_image_b64(overlay)
        except Exception as e:
            logger.debug(f"Seg overlay failed: {e}")
            return ""

    def _compute_class_distribution(self, mask: np.ndarray) -> Dict[str, float]:
        """Compute percentage of pixels per class."""
        total = mask.size
        dist = {}
        for c in range(self.num_classes):
            count = int((mask == c).sum())
            if count > 0:
                name = self.class_name_map.get(c, f"class_{c}")
                dist[name] = round(count / total * 100, 2)
        return dist

    def _identify_detected_structures(self, mask: np.ndarray) -> List[str]:
        """Return list of anatomy names present in the mask."""
        present = []
        for c in range(1, self.num_classes):
            if (mask == c).sum() > 50:
                name = self.class_name_map.get(c, f"class_{c}")
                present.append(name)
        return present

    def _aggregate_results(self, results: List[Dict], slices: List[np.ndarray]) -> Dict:
        """Aggregate multi-slice predictions into a single result."""
        if len(results) == 1:
            return results[0]

        # For disease/severity: majority vote
        disease_votes = [r["classification"]["disease_id"] for r in results]
        from collections import Counter
        disease_pred = Counter(disease_votes).most_common(1)[0][0]
        disease_conf = np.mean([r["classification"]["confidence"] for r in results])

        severity_votes = [r["severity"]["id"] for r in results]
        severity_pred = Counter(severity_votes).most_common(1)[0][0]

        # Level: union of affected levels
        all_levels = set()
        for r in results:
            all_levels.update(r["levels"]["affected"])

        # Pick most representative slice for visualization (middle)
        mid = len(results) // 2
        rep = results[mid]

        # Aggregate report from representative slice
        agg_pred = {
            "disease_pred": disease_pred,
            "disease_confidence": disease_conf,
            "severity_pred": severity_pred,
            "level_pred": [1 if l in all_levels else 0 for l in
                          ["T10/T11","T11/T12","T12/L1","L1/L2","L2/L3","L3/L4","L4/L5","L5/S1"]],
            "pfirrmann_score": np.mean([r["pfirrmann_grade"] for r in results]),
        }
        report = self.report_generator.generate(agg_pred)

        return {
            **rep,
            "classification": {
                **rep["classification"],
                "disease_id": disease_pred,
                "confidence": round(disease_conf, 4),
            },
            "severity": {"id": severity_pred, "name": ["Mild","Moderate","Severe"][severity_pred]},
            "levels": {"affected": sorted(list(all_levels)), "all_probs": rep["levels"]["all_probs"]},
            "pfirrmann_grade": round(float(agg_pred["pfirrmann_score"]), 2),
            "report": report,
            "per_slice_count": len(results),
        }
