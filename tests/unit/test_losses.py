"""
Unit tests for ATM-Net++ loss functions.
"""

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def batch_seg():
    """Synthetic (B=2, C=20, H=64, W=64) segmentation batch."""
    B, C, H, W = 2, 20, 64, 64
    logits = torch.randn(B, C, H, W)
    target = torch.randint(0, C, (B, H, W))
    return logits, target


class TestDiceLoss:
    def test_forward_shape(self, batch_seg):
        from training.losses.combined_loss import DiceLoss
        logits, target = batch_seg
        loss_fn = DiceLoss(num_classes=20, include_background=False)
        loss = loss_fn(logits, target)
        assert loss.shape == (), "DiceLoss must return scalar"

    def test_range(self, batch_seg):
        from training.losses.combined_loss import DiceLoss
        logits, target = batch_seg
        loss_fn = DiceLoss(num_classes=20)
        loss = loss_fn(logits, target)
        assert 0.0 <= loss.item() <= 2.0, f"DiceLoss out of range: {loss.item()}"

    def test_perfect_prediction_near_zero(self):
        from training.losses.combined_loss import DiceLoss
        C, H, W = 3, 16, 16
        # One-hot perfect prediction
        target = torch.zeros(1, H, W, dtype=torch.long)
        target[0, :8, :8] = 1
        target[0, 8:, 8:] = 2
        logits = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float() * 10
        loss_fn = DiceLoss(num_classes=C, include_background=True)
        loss = loss_fn(logits, target)
        assert loss.item() < 0.05, f"Perfect prediction loss too high: {loss.item()}"


class TestFocalLoss:
    def test_forward_segmentation(self, batch_seg):
        from training.losses.combined_loss import FocalLoss
        logits, target = batch_seg
        loss_fn = FocalLoss(gamma=2.0)
        loss = loss_fn(logits, target)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_forward_classification(self):
        from training.losses.combined_loss import FocalLoss
        B, C = 4, 7
        logits = torch.randn(B, C)
        target = torch.randint(0, C, (B,))
        loss_fn = FocalLoss(gamma=2.0)
        loss = loss_fn(logits, target)
        assert loss.item() >= 0


class TestCombinedLoss:
    def test_all_losses_computed(self, batch_seg):
        from training.losses.combined_loss import ATMNetLoss
        logits, seg_target = batch_seg
        B = logits.shape[0]

        loss_fn = ATMNetLoss(num_seg_classes=20)

        output = {
            "seg_logits": logits,
            "disease": {"logits": torch.randn(B, 7)},
            "severity": {"logits": torch.randn(B, 3)},
            "level": {"logits": torch.randn(B, 8)},
            "ivd_pathology": {
                "logits": torch.randn(B, 7),
                "pfirrmann_score": torch.ones(B) * 3.0,
            },
            "report_disease_logits": torch.randn(B, 7),
            "img_proj": F.normalize(torch.randn(B, 256), dim=-1),
            "txt_proj": F.normalize(torch.randn(B, 256), dim=-1),
        }
        batch = {
            "mask": seg_target,
            "disease_label": torch.randint(0, 7, (B,)),
            "severity_label": torch.randint(0, 3, (B,)),
        }

        losses = loss_fn(output, batch)
        assert "total" in losses
        assert losses["total"].item() > 0
        assert torch.isfinite(losses["total"]), "Total loss must be finite"

    def test_no_nan(self, batch_seg):
        from training.losses.combined_loss import ATMNetLoss
        logits, seg_target = batch_seg
        B = logits.shape[0]
        loss_fn = ATMNetLoss(num_seg_classes=20)
        output = {
            "seg_logits": logits,
            "disease": {"logits": torch.randn(B, 7)},
            "severity": {"logits": torch.randn(B, 3)},
            "level": {"logits": torch.randn(B, 8)},
            "ivd_pathology": {"logits": torch.randn(B, 7), "pfirrmann_score": torch.ones(B) * 3.0},
            "report_disease_logits": torch.randn(B, 7),
            "img_proj": F.normalize(torch.randn(B, 256), dim=-1),
            "txt_proj": F.normalize(torch.randn(B, 256), dim=-1),
        }
        batch = {"mask": seg_target, "disease_label": torch.zeros(B, dtype=torch.long),
                 "severity_label": torch.zeros(B, dtype=torch.long)}
        losses = loss_fn(output, batch)
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                assert not torch.isnan(v), f"NaN in loss: {k}"
