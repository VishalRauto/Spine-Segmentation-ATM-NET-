"""
Unit tests for segmentation and classification metrics.
"""

import numpy as np
import pytest
import torch


class TestDiceMetric:
    def test_perfect_dice(self):
        from training.metrics.segmentation_metrics import compute_dice
        pred = np.array([[1, 1], [2, 2]], dtype=np.int64)
        target = np.array([[1, 1], [2, 2]], dtype=np.int64)
        result = compute_dice(pred, target, num_classes=3)
        assert result["mean_dice"] > 0.99

    def test_zero_dice_no_overlap(self):
        from training.metrics.segmentation_metrics import compute_dice
        pred = np.ones((4, 4), dtype=np.int64)
        target = np.ones((4, 4), dtype=np.int64) * 2
        result = compute_dice(pred, target, num_classes=3)
        assert result.get("class_dice_1", 1.0) < 0.01 or result.get("class_dice_2", 1.0) < 0.01

    def test_partial_overlap(self):
        from training.metrics.segmentation_metrics import compute_dice
        pred = np.zeros((4, 4), dtype=np.int64)
        pred[:2, :2] = 1
        target = np.zeros((4, 4), dtype=np.int64)
        target[:2, :4] = 1  # Double the area
        result = compute_dice(pred, target, num_classes=2)
        # Intersection=4, union=8+4=12, dice=8/12 ≈ 0.667
        assert abs(result.get("class_dice_1", 0) - 0.667) < 0.05

    def test_background_excluded_by_default(self):
        from training.metrics.segmentation_metrics import compute_dice
        pred = np.zeros((8, 8), dtype=np.int64)
        target = np.zeros((8, 8), dtype=np.int64)
        # All background — no foreground classes
        result = compute_dice(pred, target, num_classes=3, include_background=False)
        # mean_dice should be 0 (no valid classes computed)
        assert result["mean_dice"] == 0.0


class TestIoUMetric:
    def test_perfect_iou(self):
        from training.metrics.segmentation_metrics import compute_iou
        pred = np.array([[1, 1], [0, 0]], dtype=np.int64)
        target = np.array([[1, 1], [0, 0]], dtype=np.int64)
        result = compute_iou(pred, target, num_classes=2)
        assert result["class_iou_1"] > 0.99

    def test_iou_less_than_dice(self):
        from training.metrics.segmentation_metrics import compute_dice, compute_iou
        pred = np.zeros((8, 8), dtype=np.int64)
        pred[:4, :4] = 1
        target = np.zeros((8, 8), dtype=np.int64)
        target[:4, :8] = 1
        dice_r = compute_dice(pred, target, num_classes=2)
        iou_r = compute_iou(pred, target, num_classes=2)
        # IoU <= Dice always
        assert iou_r.get("class_iou_1", 0) <= dice_r.get("class_dice_1", 1) + 1e-5


class TestMetricTracker:
    def test_tracker_update_and_compute(self):
        from training.metrics.segmentation_metrics import SegmentationMetricTracker
        tracker = SegmentationMetricTracker(num_classes=5)
        logits = torch.randn(2, 5, 32, 32)
        target = torch.randint(0, 5, (2, 32, 32))
        tracker.update(logits, target, compute_hd=False)
        result = tracker.compute()
        assert "mean_dice" in result
        assert "mean_iou" in result
        assert 0.0 <= result["mean_dice"] <= 1.0

    def test_tracker_reset(self):
        from training.metrics.segmentation_metrics import SegmentationMetricTracker
        tracker = SegmentationMetricTracker(num_classes=5)
        logits = torch.randn(2, 5, 32, 32)
        target = torch.randint(0, 5, (2, 32, 32))
        tracker.update(logits, target)
        tracker.reset()
        result = tracker.compute()
        assert result == {}
