"""
Multi-Task Classification Heads for ATM-Net++.

Implements:
- Disease Classification (7 classes)
- Severity Estimation (3 classes)
- Level Localization (8 IVD levels)
- Vertebra Detection
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiseaseClassificationHead(nn.Module):
    """
    Disease classification from fused multimodal features.
    7 classes: Normal, Herniation, Bulge, Stenosis, DDD, Spondylolisthesis, Fracture
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_classes: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=-1)
        confidence, pred = probs.max(dim=-1)
        return {
            "logits": logits,
            "probs": probs,
            "pred": pred,
            "confidence": confidence,
        }


class SeverityEstimationHead(nn.Module):
    """
    Severity estimation: Mild / Moderate / Severe.
    Also outputs a continuous severity score [0, 1].
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.classifier(x)
        score = self.regressor(x).squeeze(-1)
        probs = F.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)
        return {
            "logits": logits,
            "probs": probs,
            "pred": pred,
            "score": score,
        }


class LevelLocalizationHead(nn.Module):
    """
    Multi-label IVD level detection.
    Predicts presence of pathology at each of 8 lumbar IVD levels:
    T10/T11, T11/T12, T12/L1, L1/L2, L2/L3, L3/L4, L4/L5, L5/S1
    """

    LEVEL_NAMES = [
        "T10_T11", "T11_T12", "T12_L1",
        "L1_L2", "L2_L3", "L3_L4", "L4_L5", "L5_S1"
    ]

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_levels: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.detector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_levels),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.detector(x)
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).float()
        return {
            "logits": logits,
            "probs": probs,
            "pred": pred,
        }


class PerIVDPathologyHead(nn.Module):
    """
    Per-IVD multi-label pathology detection.
    Predicts multiple binary pathology labels per disc level.
    Based on SPIDER grading categories.
    """

    PATHOLOGY_NAMES = [
        "modic", "up_endplate", "low_endplate",
        "spondylolisthesis", "disc_herniation",
        "disc_narrowing", "disc_bulging"
    ]

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.num_pathologies = len(self.PATHOLOGY_NAMES)
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, self.num_pathologies),
        )
        # Pfirrmann grade regressor (1-5)
        self.pfirrmann_head = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Output [0,1], scale to [1,5] externally
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.head(x)
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).float()
        pfirrmann_score = self.pfirrmann_head(x).squeeze(-1) * 4 + 1  # Scale to [1,5]
        return {
            "logits": logits,
            "probs": probs,
            "pred": pred,
            "pfirrmann_score": pfirrmann_score,
        }


class MultiTaskHead(nn.Module):
    """
    Combined multi-task head that aggregates all classification outputs.
    Applies task-specific feature adaptation before each head.
    """

    def __init__(
        self,
        input_dim: int = 512,
        num_disease_classes: int = 7,
        num_severity_classes: int = 3,
        num_levels: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Task-specific feature adapters (lightweight)
        self.disease_adapter = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
        )
        self.severity_adapter = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
        )
        self.level_adapter = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
        )

        # Task heads
        self.disease_head = DiseaseClassificationHead(input_dim, 256, num_disease_classes, dropout)
        self.severity_head = SeverityEstimationHead(input_dim, 128, dropout)
        self.level_head = LevelLocalizationHead(input_dim, 256, num_levels, dropout)
        self.ivd_pathology_head = PerIVDPathologyHead(input_dim, 256)

    def forward(self, fused_features: torch.Tensor) -> Dict[str, Dict]:
        """
        Args:
            fused_features: (B, input_dim) multimodal fused embedding

        Returns:
            dict with task-specific output dicts
        """
        disease_feat = self.disease_adapter(fused_features)
        severity_feat = self.severity_adapter(fused_features)
        level_feat = self.level_adapter(fused_features)

        return {
            "disease": self.disease_head(disease_feat),
            "severity": self.severity_head(severity_feat),
            "level": self.level_head(level_feat),
            "ivd_pathology": self.ivd_pathology_head(fused_features),
        }
