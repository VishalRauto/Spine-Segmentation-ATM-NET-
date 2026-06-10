"""
Data augmentation pipeline for lumbar spine MRI.
Implements all augmentations required by ATM-Net++:
- Rotation, flipping, elastic deformation
- Intensity shifts, Gaussian noise
- Random crop, zoom
- Paired image+mask transforms (spatial ops remain consistent).
"""

from __future__ import annotations

import random
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


class SpineAugmentor:
    """
    Augmentation pipeline for paired (image, mask) 2D spine MRI slices.

    All spatial transforms are applied consistently to both image and mask.
    Intensity transforms are applied only to the image.
    """

    def __init__(
        self,
        rotation_range: float = 15.0,
        flip_prob: float = 0.5,
        elastic_alpha: float = 100.0,
        elastic_sigma: float = 10.0,
        intensity_shift: float = 0.1,
        intensity_scale: float = 0.1,
        gaussian_noise_std: float = 0.01,
        random_crop_size: Optional[Tuple[int, int]] = None,
        zoom_range: Tuple[float, float] = (0.9, 1.1),
        gamma_range: Tuple[float, float] = (0.7, 1.5),
        p_elastic: float = 0.3,
        p_gamma: float = 0.3,
        p_noise: float = 0.5,
        p_intensity_shift: float = 0.5,
        p_zoom: float = 0.3,
        training: bool = True,
    ):
        self.rotation_range = rotation_range
        self.flip_prob = flip_prob
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.intensity_shift = intensity_shift
        self.intensity_scale = intensity_scale
        self.gaussian_noise_std = gaussian_noise_std
        self.random_crop_size = random_crop_size
        self.zoom_range = zoom_range
        self.gamma_range = gamma_range
        self.p_elastic = p_elastic
        self.p_gamma = p_gamma
        self.p_noise = p_noise
        self.p_intensity_shift = p_intensity_shift
        self.p_zoom = p_zoom
        self.training = training

    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Apply augmentation pipeline.

        Args:
            image: (H, W) or (C, H, W) float32 array in [0,1].
            mask: (H, W) int64 array, optional.

        Returns:
            dict with keys 'image' and optionally 'mask'.
        """
        if not self.training:
            return {"image": image, "mask": mask} if mask is not None else {"image": image}

        # Unpack channel dim if present
        has_channel = image.ndim == 3
        if has_channel:
            c = image.shape[0]
            img = image[0] if c == 1 else np.transpose(image, (1, 2, 0))
        else:
            img = image.copy()

        msk = mask.copy() if mask is not None else None

        # --- Spatial Transforms ---
        img, msk = self._random_rotation(img, msk)
        img, msk = self._random_flip(img, msk)

        if random.random() < self.p_zoom:
            img, msk = self._random_zoom(img, msk)

        if random.random() < self.p_elastic:
            img, msk = self._elastic_deformation(img, msk)

        if self.random_crop_size:
            img, msk = self._random_crop(img, msk)

        # --- Intensity Transforms (image only) ---
        if random.random() < self.p_intensity_shift:
            img = self._intensity_shift(img)

        if random.random() < self.p_gamma:
            img = self._gamma_correction(img)

        if random.random() < self.p_noise:
            img = self._gaussian_noise(img)

        # Re-pack channel dim
        if has_channel:
            if c == 1:
                img = img[np.newaxis, ...]
            else:
                img = np.transpose(img, (2, 0, 1))

        result = {"image": img}
        if msk is not None:
            result["mask"] = msk
        return result

    # ------------------------------------------------------------------
    # Spatial Transforms
    # ------------------------------------------------------------------

    def _random_rotation(
        self, img: np.ndarray, msk: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        h, w = img.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        img_rot = cv2.warpAffine(img, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)
        msk_rot = None
        if msk is not None:
            msk_rot = cv2.warpAffine(msk.astype(np.float32), M, (w, h),
                                     flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0).astype(np.int64)
        return img_rot, msk_rot

    def _random_flip(
        self, img: np.ndarray, msk: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if random.random() < self.flip_prob:
            img = np.fliplr(img)
            if msk is not None:
                msk = np.fliplr(msk)
        if random.random() < self.flip_prob * 0.3:  # vertical flip less frequent
            img = np.flipud(img)
            if msk is not None:
                msk = np.flipud(msk)
        return img, msk

    def _random_zoom(
        self, img: np.ndarray, msk: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        factor = random.uniform(*self.zoom_range)
        h, w = img.shape[:2]
        new_h, new_w = int(h * factor), int(w * factor)

        # Resize
        img_zoom = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if factor > 1.0:
            # Crop center back to original
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            img_zoom = img_zoom[start_h:start_h + h, start_w:start_w + w]
        else:
            # Pad with zeros back to original
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            img_zoom = np.pad(img_zoom,
                              ((pad_h, h - new_h - pad_h),
                               (pad_w, w - new_w - pad_w)),
                              mode='constant')

        msk_zoom = None
        if msk is not None:
            msk_zoom = cv2.resize(msk.astype(np.float32), (new_w, new_h),
                                  interpolation=cv2.INTER_NEAREST).astype(np.int64)
            if factor > 1.0:
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                msk_zoom = msk_zoom[start_h:start_h + h, start_w:start_w + w]
            else:
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                msk_zoom = np.pad(msk_zoom,
                                  ((pad_h, h - new_h - pad_h),
                                   (pad_w, w - new_w - pad_w)),
                                  mode='constant').astype(np.int64)

        return img_zoom, msk_zoom

    def _elastic_deformation(
        self, img: np.ndarray, msk: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply elastic deformation (Simard et al., 2003)."""
        h, w = img.shape[:2]
        dx = gaussian_filter(
            (np.random.rand(h, w) * 2 - 1), self.elastic_sigma
        ) * self.elastic_alpha
        dy = gaussian_filter(
            (np.random.rand(h, w) * 2 - 1), self.elastic_sigma
        ) * self.elastic_alpha

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        indices = (
            np.clip(y + dy, 0, h - 1).ravel(),
            np.clip(x + dx, 0, w - 1).ravel(),
        )

        if img.ndim == 3:
            channels = [map_coordinates(img[:, :, c], indices, order=1).reshape(h, w)
                        for c in range(img.shape[2])]
            img_def = np.stack(channels, axis=2)
        else:
            img_def = map_coordinates(img, indices, order=1).reshape(h, w)

        msk_def = None
        if msk is not None:
            msk_def = map_coordinates(msk.astype(float), indices, order=0).reshape(h, w).astype(np.int64)

        return img_def.astype(np.float32), msk_def

    def _random_crop(
        self, img: np.ndarray, msk: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        crop_h, crop_w = self.random_crop_size
        h, w = img.shape[:2]

        if h < crop_h or w < crop_w:
            return img, msk

        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        img_crop = img[top:top + crop_h, left:left + crop_w]
        msk_crop = msk[top:top + crop_h, left:left + crop_w] if msk is not None else None

        # Resize back to original size
        orig_h, orig_w = h, w
        img_crop = cv2.resize(img_crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        if msk_crop is not None:
            msk_crop = cv2.resize(msk_crop.astype(np.float32), (orig_w, orig_h),
                                  interpolation=cv2.INTER_NEAREST).astype(np.int64)

        return img_crop, msk_crop

    # ------------------------------------------------------------------
    # Intensity Transforms
    # ------------------------------------------------------------------

    def _intensity_shift(self, img: np.ndarray) -> np.ndarray:
        shift = random.uniform(-self.intensity_shift, self.intensity_shift)
        scale = random.uniform(1 - self.intensity_scale, 1 + self.intensity_scale)
        return np.clip(img * scale + shift, 0.0, 1.0).astype(np.float32)

    def _gamma_correction(self, img: np.ndarray) -> np.ndarray:
        gamma = random.uniform(*self.gamma_range)
        return np.clip(np.power(img + 1e-8, gamma), 0.0, 1.0).astype(np.float32)

    def _gaussian_noise(self, img: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.gaussian_noise_std, img.shape).astype(np.float32)
        return np.clip(img + noise, 0.0, 1.0).astype(np.float32)


class TestTimeAugmentation:
    """
    Test-time augmentation (TTA) for inference.
    Applies a set of deterministic transforms and averages predictions.
    """

    TRANSFORMS = [
        {"flip_h": False, "flip_v": False},
        {"flip_h": True,  "flip_v": False},
        {"flip_h": False, "flip_v": True},
        {"flip_h": True,  "flip_v": True},
    ]

    @staticmethod
    def apply(image: np.ndarray, transform: Dict) -> np.ndarray:
        img = image.copy()
        if transform.get("flip_h"):
            img = np.flip(img, axis=-1).copy()
        if transform.get("flip_v"):
            img = np.flip(img, axis=-2).copy()
        return img

    @staticmethod
    def reverse(prediction: np.ndarray, transform: Dict) -> np.ndarray:
        pred = prediction.copy()
        if transform.get("flip_v"):
            pred = np.flip(pred, axis=-2).copy()
        if transform.get("flip_h"):
            pred = np.flip(pred, axis=-1).copy()
        return pred
