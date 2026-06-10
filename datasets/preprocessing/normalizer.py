"""
Intensity normalization and preprocessing for lumbar spine MRI.
Handles T1/T2 modality-specific normalization strategies.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np


class NormalizationStrategy(str, Enum):
    Z_SCORE = "z_score"
    MIN_MAX = "min_max"
    PERCENTILE = "percentile"
    HISTOGRAM_MATCHING = "histogram_matching"
    NYUL = "nyul"  # Nyul-style histogram normalization


class SpineMRINormalizer:
    """
    Multi-strategy normalizer for lumbar spine MRI volumes.

    Supports T1 and T2 modality-specific normalization.
    """

    def __init__(
        self,
        strategy: NormalizationStrategy = NormalizationStrategy.PERCENTILE,
        percentile_low: float = 0.5,
        percentile_high: float = 99.5,
        output_range: Tuple[float, float] = (0.0, 1.0),
    ):
        self.strategy = strategy
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.output_min, self.output_max = output_range

    def normalize(
        self,
        volume: np.ndarray,
        modality: str = "T2",
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Normalize an MRI volume.

        Args:
            volume: Input array, any shape. Float32 expected.
            modality: "T1" or "T2" (affects clip bounds).
            mask: Optional ROI mask to compute stats within.

        Returns:
            Normalized float32 array.
        """
        volume = volume.astype(np.float32)

        if self.strategy == NormalizationStrategy.Z_SCORE:
            return self._z_score(volume, mask)
        elif self.strategy == NormalizationStrategy.MIN_MAX:
            return self._min_max(volume, mask)
        elif self.strategy == NormalizationStrategy.PERCENTILE:
            return self._percentile(volume, mask)
        else:
            return self._percentile(volume, mask)

    def _z_score(self, volume: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        region = volume[mask > 0] if mask is not None else volume
        mu = float(np.mean(region))
        sigma = float(np.std(region)) + 1e-8
        normalized = (volume - mu) / sigma
        return normalized.astype(np.float32)

    def _min_max(self, volume: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        region = volume[mask > 0] if mask is not None else volume
        v_min = float(np.min(region))
        v_max = float(np.max(region)) + 1e-8
        normalized = (volume - v_min) / (v_max - v_min)
        normalized = normalized * (self.output_max - self.output_min) + self.output_min
        return np.clip(normalized, self.output_min, self.output_max).astype(np.float32)

    def _percentile(self, volume: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
        region = volume[mask > 0] if mask is not None else volume
        p_low = float(np.percentile(region, self.percentile_low))
        p_high = float(np.percentile(region, self.percentile_high))
        clipped = np.clip(volume, p_low, p_high)
        normalized = (clipped - p_low) / (p_high - p_low + 1e-8)
        normalized = normalized * (self.output_max - self.output_min) + self.output_min
        return np.clip(normalized, self.output_min, self.output_max).astype(np.float32)


class SpinePreprocessor:
    """
    Full preprocessing pipeline for a single spine MRI slice/volume.

    Steps:
    1. Convert to float32
    2. Resample to isotropic spacing (optional)
    3. Crop/pad to target size
    4. Normalize intensity
    5. Add channel dimension
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        target_spacing: Optional[Tuple[float, float]] = None,
        normalize_strategy: NormalizationStrategy = NormalizationStrategy.PERCENTILE,
        add_channel_dim: bool = True,
    ):
        self.target_size = target_size
        self.target_spacing = target_spacing
        self.normalizer = SpineMRINormalizer(strategy=normalize_strategy)
        self.add_channel_dim = add_channel_dim

    def process_slice(
        self,
        slice_2d: np.ndarray,
        modality: str = "T2",
    ) -> np.ndarray:
        """
        Process a single 2D MRI slice.

        Args:
            slice_2d: (H, W) numpy array
            modality: "T1" or "T2"

        Returns:
            (1, H, W) or (H, W) float32 array, resized and normalized.
        """
        import cv2

        img = slice_2d.astype(np.float32)

        # Resize to target
        if img.shape != self.target_size:
            img = cv2.resize(img, (self.target_size[1], self.target_size[0]),
                             interpolation=cv2.INTER_LINEAR)

        # Normalize
        img = self.normalizer.normalize(img, modality=modality)

        if self.add_channel_dim:
            img = img[np.newaxis, ...]  # (1, H, W)

        return img

    def process_mask_slice(
        self,
        mask_2d: np.ndarray,
        label_mapping: Optional[Dict[int, int]] = None,
    ) -> np.ndarray:
        """
        Process a single 2D segmentation mask.

        Args:
            mask_2d: (H, W) integer label array
            label_mapping: Optional dict to remap label values

        Returns:
            (H, W) int64 array, resized with nearest neighbor.
        """
        import cv2

        mask = mask_2d.astype(np.int32)

        if label_mapping:
            remapped = np.zeros_like(mask)
            for src, dst in label_mapping.items():
                remapped[mask == src] = dst
            mask = remapped

        if mask.shape != self.target_size:
            mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]),
                              interpolation=cv2.INTER_NEAREST)

        return mask.astype(np.int64)

    def process_volume(
        self,
        volume: np.ndarray,
        modality: str = "T2",
        slice_axis: int = 0,
    ) -> np.ndarray:
        """
        Process full 3D volume slice by slice.

        Args:
            volume: (Z, H, W) array
            modality: MRI modality
            slice_axis: axis along which to iterate (default: 0 = Z)

        Returns:
            (Z, 1, H, W) float32 array
        """
        slices = []
        for i in range(volume.shape[slice_axis]):
            s = np.take(volume, i, axis=slice_axis)
            processed = self.process_slice(s, modality=modality)
            slices.append(processed)
        return np.stack(slices, axis=0)
