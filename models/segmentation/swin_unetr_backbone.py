"""
Swin UNETR Backbone for ATM-Net++.
Extends the MONAI Swin UNETR with:
- Residual blocks in the decoder
- Attention gates at skip connections
- Deep supervision heads
- Multi-scale feature output for fusion
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    """2D Residual convolution block with GroupNorm + GELU."""

    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        groups = min(groups, out_channels)
        while out_channels % groups != 0 and groups > 1:
            groups -= 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act = nn.GELU()

        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(groups, out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + identity)


class AttentionGate(nn.Module):
    """
    Attention gate for skip connections (Oktay et al., 2018).
    Computes spatial attention from gating signal and skip connection.
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: Gating signal from deeper layer (B, F_g, H, W)
            x: Skip connection feature (B, F_l, H, W)
        Returns:
            Attended skip features (B, F_l, H, W)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=False)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DeepSupervisionHead(nn.Module):
    """Deep supervision output head at intermediate scale."""

    def __init__(self, in_channels: int, num_classes: int, scale_factor: int = 1):
        super().__init__()
        self.scale_factor = scale_factor
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.GELU(),
            nn.Conv2d(in_channels // 2, num_classes, 1),
        )

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        out = self.head(x)
        if out.shape[2:] != target_size:
            out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)
        return out


class SwinUNETRDecoder(nn.Module):
    """
    Custom decoder for Swin UNETR that adds:
    - Attention gates on skip connections
    - Residual blocks at each scale
    - Deep supervision heads
    """

    def __init__(
        self,
        num_classes: int,
        feature_size: int = 48,
        decoder_channels: Sequence[int] = (768, 384, 192, 96, 48),
        deep_supervision: bool = True,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes

        # Channel progression through decoder
        # Encoder outputs: 48, 96, 192, 384, 768 (feature_size * 2^level)
        enc_channels = [feature_size * (2**i) for i in range(5)]  # [48, 96, 192, 384, 768]

        # Decoder upsampling blocks
        self.up_blocks = nn.ModuleList()
        self.att_gates = nn.ModuleList()
        self.ds_heads = nn.ModuleList()

        in_ch = enc_channels[4]  # 768 (bottleneck)
        for i, skip_ch in enumerate(reversed(enc_channels[:4])):  # 384, 192, 96, 48
            out_ch = skip_ch
            self.up_blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                ResidualConvBlock(in_ch + skip_ch, out_ch),
                ResidualConvBlock(out_ch, out_ch),
            ))
            self.att_gates.append(AttentionGate(F_g=in_ch, F_l=skip_ch, F_int=skip_ch // 2))
            if deep_supervision and i < 3:
                self.ds_heads.append(DeepSupervisionHead(out_ch, num_classes))
            in_ch = out_ch

        # Final segmentation head
        self.seg_head = nn.Sequential(
            ResidualConvBlock(in_ch, in_ch),
            nn.Conv2d(in_ch, num_classes, 1),
        )

        # Feature projection for fusion (bottleneck features)
        self.feature_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(enc_channels[4], 768),
            nn.LayerNorm(768),
        )

    def forward(
        self,
        enc_features: List[torch.Tensor],
        return_deep_supervision: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            enc_features: List of encoder features [e0, e1, e2, e3, bottleneck]
                          Sizes go from (B,48,H,W) to (B,768,H/16,W/16)
            return_deep_supervision: Return intermediate predictions

        Returns:
            dict with 'seg_logits', 'image_features', and optionally 'ds_logits'
        """
        target_size = enc_features[0].shape[2:]  # (H, W) of original

        x = enc_features[4]  # Bottleneck: (B, 768, H/16, W/16)
        skip_features = list(reversed(enc_features[:4]))  # [e3, e2, e1, e0]

        ds_outputs = []
        for i, (up_block, att_gate, skip) in enumerate(
            zip(self.up_blocks, self.att_gates, skip_features)
        ):
            # Attend skip connection
            attended_skip = att_gate(g=x, x=skip)

            # Upsample + concat + conv
            x_up = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x_up, attended_skip], dim=1)
            # Apply residual blocks (skip upsample in up_blocks)
            x = up_block[1](x)  # ResidualConvBlock
            x = up_block[2](x)  # ResidualConvBlock

            if self.deep_supervision and i < len(self.ds_heads):
                ds_outputs.append(self.ds_heads[i](x, target_size))

        # Final segmentation
        seg_logits = F.interpolate(
            self.seg_head(x), size=target_size, mode="bilinear", align_corners=False
        )

        # Global image features for multimodal fusion
        image_features = self.feature_proj(enc_features[4])

        result = {
            "seg_logits": seg_logits,
            "image_features": image_features,
        }

        if return_deep_supervision:
            result["ds_logits"] = ds_outputs

        return result


class SpineSwinUNETR(nn.Module):
    """
    Complete Swin UNETR backbone for lumbar spine segmentation.
    Wraps MONAI's SwinUNETR encoder with ATM-Net++ custom decoder.
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (512, 512),
        in_channels: int = 1,
        num_classes: int = 20,
        feature_size: int = 48,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.1,
        use_checkpoint: bool = True,
        deep_supervision: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.feature_size = feature_size
        self.deep_supervision = deep_supervision

        # Import MONAI SwinUNETR encoder
        try:
            from monai.networks.nets import SwinUNETR as MonaiSwinUNETR
            # Use 2D spatial dims with 2D img_size
            self.encoder = MonaiSwinUNETR(
                img_size=img_size,
                in_channels=in_channels,
                out_channels=num_classes,  # We'll replace the decoder
                feature_size=feature_size,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                dropout_path_rate=dropout_path_rate,
                use_checkpoint=use_checkpoint,
                spatial_dims=2,
            )
            self._use_monai = True
        except Exception:
            self._use_monai = False
            self.encoder = self._build_fallback_encoder(in_channels, feature_size)

        # Custom decoder
        self.decoder = SwinUNETRDecoder(
            num_classes=num_classes,
            feature_size=feature_size,
            deep_supervision=deep_supervision,
        )

    def _build_fallback_encoder(self, in_channels: int, feature_size: int) -> nn.Module:
        """Simple CNN encoder fallback if MONAI is not available."""
        return nn.Sequential(
            nn.Conv2d(in_channels, feature_size, 7, 2, 3, bias=False),
            nn.BatchNorm2d(feature_size),
            nn.GELU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) input MRI
            return_features: Also return multi-scale encoder features

        Returns:
            dict with 'seg_logits', 'image_features', optionally 'ds_logits'
        """
        if self._use_monai:
            # Extract encoder features using MONAI SwinUNETR internals
            enc_features = self._extract_monai_features(x)
        else:
            enc_features = self._extract_fallback_features(x)

        decoder_out = self.decoder(
            enc_features,
            return_deep_supervision=self.deep_supervision and self.training,
        )

        if return_features:
            decoder_out["enc_features"] = enc_features

        return decoder_out

    def _extract_monai_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract hierarchical features from MONAI SwinUNETR encoder.
        Returns 5 feature maps at different scales.
        """
        # Access internal MONAI SwinTransformer
        swin = self.encoder.swinViT
        B, C, H, W = x.shape

        # Patch embedding
        x_patch = swin.patch_embed(x)
        if swin.ape:
            x_patch = x_patch + swin.absolute_pos_embed
        x_patch = swin.pos_drop(x_patch)

        # Extract features at each stage
        features = []
        x_stage = x_patch
        for layer in swin.layers:
            x_stage = layer(x_stage)
            # Reshape to spatial: (B, H'*W', C) -> (B, C, H', W')
            feat = x_stage
            Wh = H // (2 ** (len(features) + 2))
            Ww = W // (2 ** (len(features) + 2))
            try:
                feat_spatial = feat.view(B, Wh, Ww, -1).permute(0, 3, 1, 2).contiguous()
            except Exception:
                # Fallback: adaptive pooling
                feat_spatial = feat.mean(dim=1, keepdim=True).expand(B, self.feature_size * (2**len(features)), 1, 1)
            features.append(feat_spatial)

        # Pad to 5 features if needed
        while len(features) < 5:
            features.append(features[-1])

        # Also add stem output (full resolution features)
        try:
            enc0 = self.encoder.encoder1(x)    # (B, F, H, W)
            enc1 = self.encoder.encoder2(x)    # (B, 2F, H/2, W/2)
            enc2 = self.encoder.encoder3(x)    # (B, 4F, H/4, W/4)
            enc3 = self.encoder.encoder4(x)    # (B, 8F, H/8, W/8)
            enc4 = self.encoder.encoder10(x)   # (B, 16F, H/16, W/16)
            return [enc0, enc1, enc2, enc3, enc4]
        except Exception:
            return features[:5]

    def _extract_fallback_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Simple U-Net style feature extraction for fallback."""
        fs = self.feature_size
        feats = []
        for i in range(5):
            stride = 2 ** i
            out_ch = fs * (2 ** i)
            conv = nn.Conv2d(
                x.shape[1] if i == 0 else fs * (2 ** (i - 1)),
                out_ch, 3, stride if i > 0 else 1, 1
            ).to(x.device)
            x = F.relu(conv(x))
            feats.append(x)
        return feats


class SegmentationHead(nn.Module):
    """Final segmentation output with optional CRF post-processing."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
