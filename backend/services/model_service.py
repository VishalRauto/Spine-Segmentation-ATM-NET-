"""
Model service: singleton model loader and inference dispatcher.
Thread-safe lazy initialization with health check.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

logger = logging.getLogger(__name__)

_predictor_instance: Optional[Any] = None
_lock = asyncio.Lock()


async def get_predictor():
    """
    Return the singleton SpinePredictor.
    Lazy-loaded on first call, thread-safe.
    """
    global _predictor_instance
    if _predictor_instance is not None:
        return _predictor_instance

    async with _lock:
        if _predictor_instance is not None:
            return _predictor_instance

        from backend.core.config import get_settings
        settings = get_settings()

        config_path = Path(settings.MODEL_CONFIG_PATH)
        if not config_path.exists():
            logger.error(f"Config not found: {config_path}")
            _predictor_instance = _DummyPredictor()
            return _predictor_instance

        with open(config_path) as f:
            config = yaml.safe_load(f)

        device = torch.device(settings.MODEL_DEVICE)
        checkpoint_path = Path(settings.MODEL_CHECKPOINT_PATH)

        if not checkpoint_path.exists():
            logger.warning(
                f"Checkpoint not found: {checkpoint_path}. "
                "Using untrained model for demonstration."
            )
            from models.atmnet_plus_plus import ATMNetPlusPlus
            model_cfg = config.get("model", {})
            model = ATMNetPlusPlus(
                img_size=tuple(model_cfg.get("img_size", [512, 512])),
                in_channels=model_cfg.get("in_channels", 1),
                num_seg_classes=config.get("segmentation", {}).get("num_classes", 20),
                use_text=True,
                use_demographics=True,
            )
            from inference.predictor import SpinePredictor
            _predictor_instance = SpinePredictor(
                model=model,
                device=device,
                config=config,
                use_tta=settings.USE_TTA,
            )
        else:
            from inference.predictor import SpinePredictor
            _predictor_instance = SpinePredictor.from_checkpoint(
                checkpoint_path=str(checkpoint_path),
                config=config,
                device=device,
            )

        logger.info(f"Model loaded on {device}")
        return _predictor_instance


class _DummyPredictor:
    """Returns mock predictions when model is unavailable (dev/test mode)."""

    def predict_from_file(self, image_path: str, **kwargs) -> Dict:
        return self._mock_result()

    def predict_slice(self, image_array, **kwargs) -> Dict:
        return self._mock_result()

    def _mock_result(self) -> Dict:
        import numpy as np
        return {
            "segmentation": {
                "mask": [],
                "overlay_b64": "",
                "class_distribution": {"background": 85.0, "L4": 5.0, "L5": 4.0, "L4_L5_disc": 3.0, "L5_S1_disc": 3.0},
                "detected_structures": ["L4", "L5", "L4_L5_disc", "L5_S1_disc"],
            },
            "classification": {
                "disease_id": 2,
                "disease_name": "Disc_Bulge",
                "confidence": 0.78,
                "disease_probabilities": {
                    "Normal": 0.05, "Disc_Herniation": 0.10, "Disc_Bulge": 0.78,
                    "Spinal_Stenosis": 0.03, "Degenerative_Disc_Disease": 0.03,
                    "Spondylolisthesis": 0.01, "Compression_Fracture": 0.00,
                },
            },
            "severity": {"id": 1, "name": "Moderate"},
            "levels": {
                "affected": ["L4/L5", "L5/S1"],
                "all_probs": {"L4/L5": 0.82, "L5/S1": 0.67},
            },
            "pfirrmann_grade": 3.2,
            "report": {
                "report_text": "DEMO: Moderate disc bulging at L4/L5 and L5/S1.",
                "findings": "Moderate disc bulging at L4/L5 and L5/S1.",
                "impression": "Disc Bulge at L4/L5, L5/S1. Severity: Moderate.",
                "recommendation": "Conservative management with physical therapy.",
                "disease_name": "Disc_Bulge",
                "severity": "moderate",
                "affected_levels": ["L4/L5", "L5/S1"],
                "confidence": 0.78,
                "pfirrmann_grade": 3.2,
            },
            "gradcam_b64": "",
            "inference_time_ms": 0,
            "num_slices_processed": 0,
            "is_demo": True,
        }
