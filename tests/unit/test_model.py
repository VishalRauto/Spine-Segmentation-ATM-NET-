"""
Unit tests for ATM-Net++ model components.
Tests run on CPU with small tensors for fast CI execution.
"""

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def small_image():
    return torch.randn(1, 1, 128, 128)  # (B, C, H, W)


@pytest.fixture
def demo_features():
    return torch.rand(1, 8)


class TestLightweightUNet:
    """Test the fallback CNN encoder without MONAI dependency."""

    def test_output_shape(self, small_image):
        from models.segmentation.swin_unetr_backbone import ResidualConvBlock
        # Build a simple 2-layer conv
        net = torch.nn.Sequential(
            ResidualConvBlock(1, 16),
            ResidualConvBlock(16, 20),
        )
        out = net(small_image)
        assert out.shape == (1, 20, 128, 128)

    def test_residual_block_identity(self):
        from models.segmentation.swin_unetr_backbone import ResidualConvBlock
        block = ResidualConvBlock(32, 32)
        x = torch.randn(1, 32, 16, 16)
        out = block(x)
        assert out.shape == x.shape

    def test_attention_gate(self):
        from models.segmentation.swin_unetr_backbone import AttentionGate
        ag = AttentionGate(F_g=64, F_l=64, F_int=32)
        g = torch.randn(2, 64, 8, 8)
        x = torch.randn(2, 64, 8, 8)
        out = ag(g, x)
        assert out.shape == x.shape
        # Output should be between 0 and the input (attention gate)
        assert (out >= 0).all()


class TestDemographicEncoder:
    def test_forward(self):
        from models.fusion.multimodal_fusion import DemographicEncoder
        enc = DemographicEncoder(input_dim=8, hidden_dims=[32, 64], output_dim=64)
        x = torch.rand(4, 8)
        out = enc(x)
        assert out.shape == (4, 64)

    def test_no_nan(self):
        from models.fusion.multimodal_fusion import DemographicEncoder
        enc = DemographicEncoder(input_dim=8, hidden_dims=[32, 64], output_dim=64)
        x = torch.rand(2, 8)
        out = enc(x)
        assert not torch.isnan(out).any()


class TestMultimodalFusion:
    def test_hasf_forward(self):
        from models.fusion.multimodal_fusion import HASFModule
        hasf = HASFModule(image_dim=128, text_dim=128, demo_dim=64, fusion_dim=64)
        img = torch.randn(2, 128)
        txt = torch.randn(2, 128)
        demo = torch.randn(2, 64)
        out = hasf(img, txt, demo)
        assert out.shape == (2, 64)

    def test_atpg_forward(self):
        from models.fusion.multimodal_fusion import ATPGModule
        atpg = ATPGModule(image_dim=64, text_dim=64, num_prompts=8)
        img = torch.randn(2, 64)
        out = atpg(img)
        assert out.shape == (2, 8, 64)

    def test_cross_modal_attention(self):
        from models.fusion.multimodal_fusion import CrossModalAttention
        attn = CrossModalAttention(query_dim=64, kv_dim=64, num_heads=4)
        q = torch.randn(2, 5, 64)
        kv = torch.randn(2, 10, 64)
        out, weights = attn(q, kv)
        assert out.shape == (2, 5, 64)
        assert weights.shape == (2, 4, 5, 10)


class TestClassificationHeads:
    def test_disease_head(self):
        from models.classification.disease_classifier import DiseaseClassificationHead
        head = DiseaseClassificationHead(input_dim=64, hidden_dim=32, num_classes=7)
        x = torch.randn(4, 64)
        out = head(x)
        assert out["logits"].shape == (4, 7)
        assert out["probs"].shape == (4, 7)
        assert out["pred"].shape == (4,)
        assert (out["probs"].sum(dim=-1) - 1.0).abs().max() < 1e-5  # Softmax sums to 1

    def test_severity_head(self):
        from models.classification.disease_classifier import SeverityEstimationHead
        head = SeverityEstimationHead(input_dim=64)
        x = torch.randn(3, 64)
        out = head(x)
        assert out["logits"].shape == (3, 3)
        assert out["score"].shape == (3,)
        assert (out["score"] >= 0).all() and (out["score"] <= 1).all()

    def test_level_head(self):
        from models.classification.disease_classifier import LevelLocalizationHead
        head = LevelLocalizationHead(input_dim=64, num_levels=8)
        x = torch.randn(2, 64)
        out = head(x)
        assert out["logits"].shape == (2, 8)
        assert out["pred"].shape == (2, 8)

    def test_multitask_head(self):
        from models.classification.disease_classifier import MultiTaskHead
        head = MultiTaskHead(input_dim=64)
        x = torch.randn(2, 64)
        out = head(x)
        assert "disease" in out
        assert "severity" in out
        assert "level" in out
        assert "ivd_pathology" in out


class TestFullModel:
    """Integration test: full forward pass of ATMNetPlusPlus (CPU, small)."""

    def test_forward_image_only(self, small_image):
        from models.atmnet_plus_plus import ATMNetPlusPlus
        model = ATMNetPlusPlus(
            img_size=(128, 128),
            in_channels=1,
            num_seg_classes=5,
            feature_size=16,
            fusion_dim=64,
            use_text=False,
            use_demographics=True,
            deep_supervision=False,
        )
        model.eval()
        with torch.no_grad():
            out = model(small_image)
        assert "seg_logits" in out
        assert out["seg_logits"].shape == (1, 5, 128, 128)
        assert "disease" in out
        assert "severity" in out

    def test_forward_with_demographics(self, small_image, demo_features):
        from models.atmnet_plus_plus import ATMNetPlusPlus
        model = ATMNetPlusPlus(
            img_size=(128, 128),
            in_channels=1,
            num_seg_classes=5,
            feature_size=16,
            fusion_dim=64,
            use_text=False,
            use_demographics=True,
            deep_supervision=False,
        )
        model.eval()
        with torch.no_grad():
            out = model(small_image, demographics=demo_features)
        assert out["seg_logits"].shape == (1, 5, 128, 128)
        assert not torch.isnan(out["seg_logits"]).any()

    def test_no_nan_outputs(self, small_image):
        from models.atmnet_plus_plus import ATMNetPlusPlus
        model = ATMNetPlusPlus(
            img_size=(128, 128),
            in_channels=1,
            num_seg_classes=5,
            feature_size=16,
            fusion_dim=64,
            use_text=False,
            deep_supervision=False,
        )
        model.eval()
        with torch.no_grad():
            out = model(small_image)
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                assert not torch.isnan(v).any(), f"NaN in output: {k}"
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, torch.Tensor):
                        assert not torch.isnan(vv).any(), f"NaN in output[{k}][{kk}]"

    def test_predict_method(self, small_image):
        from models.atmnet_plus_plus import ATMNetPlusPlus
        model = ATMNetPlusPlus(
            img_size=(128, 128),
            in_channels=1,
            num_seg_classes=5,
            feature_size=16,
            fusion_dim=64,
            use_text=False,
            deep_supervision=False,
        )
        result = model.predict(small_image)
        assert "seg_pred" in result
        assert result["seg_pred"].shape == (1, 128, 128)
