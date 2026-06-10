"""
Unit tests for preprocessing pipeline.
"""

import numpy as np
import pytest


class TestNormalizer:
    def test_percentile_range(self):
        from datasets.preprocessing.normalizer import SpineMRINormalizer, NormalizationStrategy
        norm = SpineMRINormalizer(NormalizationStrategy.PERCENTILE)
        arr = np.random.uniform(0, 1000, (64, 64)).astype(np.float32)
        out = norm.normalize(arr)
        assert out.min() >= -0.01
        assert out.max() <= 1.01

    def test_zscore_mean_zero(self):
        from datasets.preprocessing.normalizer import SpineMRINormalizer, NormalizationStrategy
        norm = SpineMRINormalizer(NormalizationStrategy.Z_SCORE)
        arr = np.random.uniform(100, 500, (64, 64)).astype(np.float32)
        out = norm.normalize(arr)
        assert abs(out.mean()) < 0.1

    def test_minmax_range(self):
        from datasets.preprocessing.normalizer import SpineMRINormalizer, NormalizationStrategy
        norm = SpineMRINormalizer(NormalizationStrategy.MIN_MAX)
        arr = np.array([[0, 50, 100], [200, 500, 1000]], dtype=np.float32)
        out = norm.normalize(arr)
        assert abs(out.min()) < 0.01
        assert abs(out.max() - 1.0) < 0.01


class TestSpinePreprocessor:
    def test_process_slice_shape(self):
        from datasets.preprocessing.normalizer import SpinePreprocessor
        proc = SpinePreprocessor(target_size=(128, 128))
        img = np.random.uniform(0, 1000, (256, 200)).astype(np.float32)
        out = proc.process_slice(img)
        assert out.shape == (1, 128, 128)

    def test_process_mask_slice(self):
        from datasets.preprocessing.normalizer import SpinePreprocessor
        proc = SpinePreprocessor(target_size=(128, 128))
        mask = np.random.randint(0, 20, (256, 200)).astype(np.int64)
        out = proc.process_mask_slice(mask)
        assert out.shape == (128, 128)
        assert out.dtype == np.int64

    def test_label_values_preserved(self):
        from datasets.preprocessing.normalizer import SpinePreprocessor
        proc = SpinePreprocessor(target_size=(64, 64))
        mask = np.zeros((64, 64), dtype=np.int64)
        mask[10:20, 10:20] = 5
        out = proc.process_mask_slice(mask)
        # Label 5 should still be present after resize
        assert 5 in out


class TestLabelMapper:
    def test_remap_vertebra(self):
        from datasets.preprocessing.label_mapper import remap_spider_mask, SPIDER_TO_ATMNET
        mask = np.zeros((10, 10), dtype=np.int32)
        mask[2:5, 2:5] = 20  # L1 in SPIDER = class 4 in ATM-Net++
        out = remap_spider_mask(mask)
        assert out[3, 3] == SPIDER_TO_ATMNET[20]

    def test_remap_disc(self):
        from datasets.preprocessing.label_mapper import remap_spider_mask
        mask = np.zeros((10, 10), dtype=np.int32)
        mask[5:8, 5:8] = 122  # L4/L5 disc
        out = remap_spider_mask(mask)
        assert out[6, 6] == 16  # ATM-Net++ class 16

    def test_background_zero(self):
        from datasets.preprocessing.label_mapper import remap_spider_mask
        mask = np.zeros((8, 8), dtype=np.int32)
        out = remap_spider_mask(mask)
        assert (out == 0).all()

    def test_colorize_mask_shape(self):
        from datasets.preprocessing.label_mapper import create_colorized_mask
        mask = np.random.randint(0, 20, (64, 64), dtype=np.int64)
        rgb = create_colorized_mask(mask)
        assert rgb.shape == (64, 64, 3)
        assert rgb.dtype == np.uint8


class TestAugmentation:
    def test_augmentor_image_mask_same_shape(self):
        from datasets.transforms.augmentations import SpineAugmentor
        aug = SpineAugmentor(training=True, p_elastic=0.0)
        img = np.random.rand(128, 128).astype(np.float32)
        mask = np.random.randint(0, 5, (128, 128)).astype(np.int64)
        out = aug(img, mask)
        assert out["image"].shape == img.shape
        assert out["mask"].shape == mask.shape

    def test_augmentor_preserves_dtype(self):
        from datasets.transforms.augmentations import SpineAugmentor
        aug = SpineAugmentor(training=True, p_elastic=0.0)
        img = np.random.rand(64, 64).astype(np.float32)
        out = aug(img)
        assert out["image"].dtype == np.float32

    def test_no_augmentation_in_eval(self):
        from datasets.transforms.augmentations import SpineAugmentor
        aug = SpineAugmentor(training=False)
        img = np.random.rand(64, 64).astype(np.float32)
        mask = np.random.randint(0, 3, (64, 64)).astype(np.int64)
        out = aug(img, mask)
        np.testing.assert_array_equal(out["image"], img)
