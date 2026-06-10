"""
Multimodal Fusion Module for ATM-Net++.

Fuses:
1. Image features from Swin UNETR (768-dim)
2. Text features from Bio-ClinicalBERT (768-dim)
3. Demographic features from MLP encoder (256-dim)

Using:
- Multi-Head Cross-Attention (MHCA)
- Transformer Fusion Layers
- Feature alignment with contrastive-style projection
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DemographicEncoder(nn.Module):
    """
    MLP encoder for patient demographics and acquisition parameters.
    Input: [sex, age, num_vertebrae, num_discs, field_strength,
            pixel_spacing, echo_time, repetition_time] (8-dim)
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dims: List[int] = [64, 128, 256],
        output_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        in_d = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_d, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_d = h
        layers += [nn.Linear(in_d, output_dim), nn.LayerNorm(output_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) demographic features
        Returns:
            (B, output_dim) demographic embedding
        """
        return self.mlp(x)


class CrossModalAttention(nn.Module):
    """
    Multi-Head Cross-Attention between two modalities.
    Query from one modality, Key/Value from another.
    """

    def __init__(self, query_dim: int, kv_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert query_dim % num_heads == 0 or query_dim >= num_heads, \
            f"query_dim {query_dim} must be divisible by num_heads {num_heads}"

        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(kv_dim, query_dim)
        self.v_proj = nn.Linear(kv_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(query_dim)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (B, Nq, query_dim)
            key_value: (B, Nkv, kv_dim)
            key_padding_mask: (B, Nkv) bool mask (True = ignore)
        Returns:
            attended: (B, Nq, query_dim)
            attention_weights: (B, num_heads, Nq, Nkv)
        """
        B, Nq, _ = query.shape
        Nkv = key_value.shape[1]

        Q = self.q_proj(query).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key_value).view(B, Nkv, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(key_value).view(B, Nkv, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, Nq, -1)
        out = self.out_proj(out)

        # Residual + norm
        out = self.norm(out + query)
        return out, attn_weights


class TransformerFusionLayer(nn.Module):
    """
    Single Transformer fusion layer that processes concatenated multimodal tokens.
    Self-attention over all tokens + FFN.
    """

    def __init__(self, d_model: int, num_heads: int = 8, ffn_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model) multimodal token sequence
        Returns:
            (B, N, d_model) fused tokens
        """
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class ATPGModule(nn.Module):
    """
    Anatomy-Text Prompt Generation (ATPG) Module.
    Generates anatomy-aware text prompts from image features to guide
    text feature extraction.
    Inspired by ATM-Net ATPG component.
    """

    def __init__(self, image_dim: int = 768, text_dim: int = 768, num_prompts: int = 16):
        super().__init__()
        self.num_prompts = num_prompts
        # Generate soft prompt tokens from image features
        self.prompt_gen = nn.Sequential(
            nn.Linear(image_dim, image_dim),
            nn.GELU(),
            nn.Linear(image_dim, num_prompts * text_dim),
        )
        self.prompt_norm = nn.LayerNorm(text_dim)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_features: (B, image_dim)
        Returns:
            prompt_tokens: (B, num_prompts, text_dim)
        """
        B = image_features.shape[0]
        prompts = self.prompt_gen(image_features)
        prompts = prompts.view(B, self.num_prompts, -1)
        return self.prompt_norm(prompts)


class HASFModule(nn.Module):
    """
    Hierarchical Anatomy-aware Semantic Fusion (HASF) Module.
    Performs coarse-to-fine fusion of image, text and demographic features
    at multiple semantic levels.
    """

    def __init__(
        self,
        image_dim: int = 768,
        text_dim: int = 768,
        demo_dim: int = 256,
        fusion_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Project all modalities to fusion_dim
        self.image_proj = nn.Sequential(nn.Linear(image_dim, fusion_dim), nn.LayerNorm(fusion_dim))
        self.text_proj  = nn.Sequential(nn.Linear(text_dim, fusion_dim), nn.LayerNorm(fusion_dim))
        self.demo_proj  = nn.Sequential(nn.Linear(demo_dim, fusion_dim), nn.LayerNorm(fusion_dim))

        # Cross-attention: image attends to text
        self.img_text_cross_attn = CrossModalAttention(fusion_dim, fusion_dim, num_heads, dropout)
        # Cross-attention: image attends to demographics
        self.img_demo_cross_attn = CrossModalAttention(fusion_dim, fusion_dim, num_heads, dropout)

        # Coarse-level gating
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.Sigmoid(),
        )

        self.final_norm = nn.LayerNorm(fusion_dim)

    def forward(
        self,
        image_feat: torch.Tensor,
        text_feat: torch.Tensor,
        demo_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image_feat: (B, image_dim)
            text_feat:  (B, text_dim)
            demo_feat:  (B, demo_dim)
        Returns:
            fused: (B, fusion_dim)
        """
        # Project to common dim
        img = self.image_proj(image_feat).unsqueeze(1)   # (B, 1, D)
        txt = self.text_proj(text_feat).unsqueeze(1)     # (B, 1, D)
        demo = self.demo_proj(demo_feat).unsqueeze(1)    # (B, 1, D)

        # Cross-attention
        img_txt, _ = self.img_text_cross_attn(img, txt)   # (B, 1, D)
        img_demo, _ = self.img_demo_cross_attn(img, demo) # (B, 1, D)

        # Squeeze
        img_sq = img.squeeze(1)
        img_txt_sq = img_txt.squeeze(1)
        img_demo_sq = img_demo.squeeze(1)

        # Gated fusion
        concat = torch.cat([img_sq, img_txt_sq, img_demo_sq], dim=-1)
        gate = self.gate(concat)
        fused = gate * img_txt_sq + (1 - gate) * img_demo_sq + img_sq

        return self.final_norm(fused)


class CCAEModule(nn.Module):
    """
    Cross-modal Context-Aware Enhancement (CCAE) Module.
    Enhances image segmentation features using text context.
    Produces a context vector that modulates decoder features.
    """

    def __init__(self, fusion_dim: int = 512, spatial_channels: int = 256):
        super().__init__()
        # Generate spatial modulation parameters (scale + shift = FiLM conditioning)
        self.scale_net = nn.Sequential(
            nn.Linear(fusion_dim, spatial_channels),
            nn.GELU(),
            nn.Linear(spatial_channels, spatial_channels),
        )
        self.shift_net = nn.Sequential(
            nn.Linear(fusion_dim, spatial_channels),
            nn.GELU(),
            nn.Linear(spatial_channels, spatial_channels),
        )
        self.norm = nn.LayerNorm(spatial_channels)

    def forward(
        self,
        spatial_feat: torch.Tensor,
        context_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            spatial_feat: (B, C, H, W) spatial image features
            context_feat: (B, fusion_dim) multimodal context
        Returns:
            modulated: (B, C, H, W)
        """
        B, C, H, W = spatial_feat.shape

        scale = self.scale_net(context_feat).view(B, C, 1, 1)
        shift = self.shift_net(context_feat).view(B, C, 1, 1)

        return spatial_feat * (1 + scale) + shift


class MultimodalFusionModule(nn.Module):
    """
    Complete multimodal fusion module for ATM-Net++.

    Integrates:
    - ATPG (Anatomy-Text Prompt Generation)
    - HASF (Hierarchical Anatomy-aware Semantic Fusion)
    - CCAE (Cross-modal Context-Aware Enhancement)
    - Transformer Fusion Layers
    """

    def __init__(
        self,
        image_feat_dim: int = 768,
        text_feat_dim: int = 768,
        demo_feat_dim: int = 256,
        fusion_dim: int = 512,
        num_heads: int = 8,
        num_transformer_layers: int = 4,
        dropout: float = 0.1,
        num_atpg_prompts: int = 16,
    ):
        super().__init__()
        self.fusion_dim = fusion_dim

        # Demographic encoder
        self.demo_encoder = DemographicEncoder(
            input_dim=demo_feat_dim if demo_feat_dim <= 8 else 8,
            hidden_dims=[64, 128, 256],
            output_dim=demo_feat_dim,
            dropout=dropout,
        )

        # ATPG: anatomy-guided text prompts
        self.atpg = ATPGModule(
            image_dim=image_feat_dim,
            text_dim=text_feat_dim,
            num_prompts=num_atpg_prompts,
        )

        # HASF: hierarchical cross-modal fusion
        self.hasf = HASFModule(
            image_dim=image_feat_dim,
            text_dim=text_feat_dim,
            demo_dim=demo_feat_dim,
            fusion_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Transformer fusion over all tokens
        self.transformer_layers = nn.ModuleList([
            TransformerFusionLayer(
                d_model=fusion_dim,
                num_heads=num_heads,
                ffn_dim=fusion_dim * 4,
                dropout=dropout,
            )
            for _ in range(num_transformer_layers)
        ])

        # Output head
        self.output_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        # CCAE for spatial feature modulation
        self.ccae = CCAEModule(fusion_dim=fusion_dim, spatial_channels=256)

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        demographics: torch.Tensor,
        text_token_embeddings: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            image_features: (B, image_feat_dim) global image embedding
            text_features: (B, text_feat_dim) CLS text embedding
            demographics: (B, 8) raw demographic vector
            text_token_embeddings: (B, seq_len, text_feat_dim) optional token embeddings

        Returns:
            dict with:
                'fused_features': (B, fusion_dim) - main fused embedding
                'atpg_prompts': (B, num_prompts, text_feat_dim)
                'attention_weights': attention for visualization
        """
        # Encode demographics
        demo_feat = self.demo_encoder(demographics)  # (B, demo_feat_dim)

        # Generate anatomy-text prompts
        atpg_prompts = self.atpg(image_features)  # (B, num_prompts, text_feat_dim)

        # HASF: hierarchical fusion
        fused = self.hasf(image_features, text_features, demo_feat)  # (B, fusion_dim)

        # Transformer over [fused, text_tokens] if available
        if text_token_embeddings is not None:
            # Project text tokens to fusion_dim
            txt_proj = nn.functional.linear(
                text_token_embeddings,
                weight=torch.eye(
                    min(text_token_embeddings.shape[-1], self.fusion_dim),
                    text_token_embeddings.shape[-1],
                    device=text_token_embeddings.device
                )
            )[:, :, :self.fusion_dim]

            tokens = torch.cat([fused.unsqueeze(1), txt_proj], dim=1)
        else:
            tokens = fused.unsqueeze(1)  # (B, 1, fusion_dim)

        for layer in self.transformer_layers:
            tokens = layer(tokens)

        # Take fused token (first position)
        fused_out = tokens[:, 0, :]
        fused_out = self.output_proj(fused_out)

        return {
            "fused_features": fused_out,
            "atpg_prompts": atpg_prompts,
            "demo_features": demo_feat,
        }
