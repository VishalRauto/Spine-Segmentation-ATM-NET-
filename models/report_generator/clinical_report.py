"""
Clinical Report Generator for ATM-Net++.
Produces radiologist-style structured findings from model predictions.
"""

from __future__ import annotations

import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Template-based report generation (primary, always available)
# ─────────────────────────────────────────────────────────────────────

DISEASE_NAMES = {
    0: "Normal",
    1: "Disc Herniation",
    2: "Disc Bulge",
    3: "Spinal Stenosis",
    4: "Degenerative Disc Disease",
    5: "Spondylolisthesis",
    6: "Compression Fracture",
}

SEVERITY_NAMES = {0: "mild", 1: "moderate", 2: "severe"}

LEVEL_NAMES = [
    "T10/T11", "T11/T12", "T12/L1",
    "L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"
]

PATHOLOGY_FINDINGS_TEMPLATES = {
    1: (
        "{severity} disc herniation at {levels}. "
        "There is posterior displacement of disc material with potential neural compression."
    ),
    2: (
        "{severity} disc bulging noted at {levels}. "
        "Broad-based extension of disc beyond the vertebral endplates without frank herniation."
    ),
    3: (
        "{severity} spinal canal stenosis identified at {levels}. "
        "Reduction in the central spinal canal diameter with possible cord/cauda equina compromise."
    ),
    4: (
        "{severity} degenerative disc disease involving {levels}. "
        "Loss of disc height and signal intensity consistent with dehydration."
    ),
    5: (
        "{severity} spondylolisthesis at {levels}. "
        "Anterior displacement of the vertebral body relative to the subjacent segment."
    ),
    6: (
        "{severity} compression fracture identified. "
        "Loss of vertebral body height with endplate irregularity."
    ),
    0: (
        "No significant disc pathology identified. "
        "Normal disc height, signal intensity, and alignment throughout the examined levels."
    ),
}

RECOMMENDATIONS = {
    0: "Routine follow-up. No immediate intervention required.",
    1: "Clinical correlation recommended. Consider MRI with contrast for nerve root evaluation. "
       "Neurosurgical or orthopedic consultation may be warranted if symptomatic.",
    2: "Conservative management with physical therapy. "
       "Re-imaging in 6 months if symptoms persist or worsen.",
    3: "Neurosurgical consultation recommended for symptomatic cases. "
       "Consider decompressive surgery evaluation based on clinical presentation.",
    4: "Physical therapy, pain management, and activity modification. "
       "Repeat imaging in 12 months to monitor progression.",
    5: "Orthopedic/neurosurgical consultation. "
       "Evaluate for surgical stabilization depending on grade and symptoms.",
    6: "Urgent orthopedic evaluation. DEXA scan to assess bone density. "
       "Evaluate for osteoporosis treatment protocol.",
}


class TemplateReportGenerator:
    """
    Produces structured clinical reports from prediction dictionaries.
    Template-based, no ML required.
    """

    def generate(
        self,
        predictions: Dict,
        patient_info: Optional[Dict] = None,
        study_info: Optional[Dict] = None,
    ) -> Dict[str, str]:
        """
        Args:
            predictions: Output from ATMNetPlusPlus.forward() including
                         disease, severity, level, seg outputs.
            patient_info: Optional dict with patient demographics.
            study_info: Optional dict with study metadata.

        Returns:
            dict with 'report_text', 'findings', 'impression', 'recommendation'
        """
        # Extract prediction values
        disease_pred = int(predictions.get("disease_pred", 0))
        severity_pred = int(predictions.get("severity_pred", 0))
        confidence = float(predictions.get("disease_confidence", 0.0))
        level_preds = predictions.get("level_pred", [])
        pfirrmann = float(predictions.get("pfirrmann_score", 3.0))

        # Identify affected levels
        affected_levels = [
            LEVEL_NAMES[i]
            for i, active in enumerate(level_preds)
            if active and i < len(LEVEL_NAMES)
        ]
        if not affected_levels:
            affected_levels = ["the lumbar spine"]

        levels_str = ", ".join(affected_levels)
        severity_str = SEVERITY_NAMES.get(severity_pred, "mild")
        disease_name = DISEASE_NAMES.get(disease_pred, "Unknown")

        # Build findings section
        template = PATHOLOGY_FINDINGS_TEMPLATES.get(disease_pred,
                   PATHOLOGY_FINDINGS_TEMPLATES[0])
        findings_text = template.format(
            severity=severity_str.capitalize(),
            levels=levels_str,
        )

        # Additional grading note
        if pfirrmann > 1:
            findings_text += (
                f" Mean Pfirrmann grade of {pfirrmann:.1f} indicates "
                f"{'early' if pfirrmann < 3 else ('moderate' if pfirrmann < 4 else 'advanced')} "
                f"disc degeneration."
            )

        # Impression
        impression = (
            f"{disease_name}"
            + (f" at {levels_str}" if disease_pred != 0 else "")
            + f". Severity: {severity_str.capitalize()}."
            + f" Diagnostic confidence: {confidence * 100:.1f}%."
        )

        # Recommendation
        recommendation = RECOMMENDATIONS.get(disease_pred, RECOMMENDATIONS[0])

        # Full report
        now = datetime.datetime.now()
        patient_str = ""
        if patient_info:
            age = patient_info.get("age", "N/A")
            sex = patient_info.get("sex", "N/A")
            patient_str = f"Patient: {sex}, Age {age}\n"

        report_text = (
            f"LUMBAR SPINE MRI REPORT\n"
            f"{'=' * 40}\n"
            f"Date: {now.strftime('%Y-%m-%d %H:%M')}\n"
            f"{patient_str}"
            f"\nTECHNIQUE:\n"
            f"Sagittal T1 and T2-weighted sequences of the lumbar spine.\n"
            f"\nFINDINGS:\n"
            f"{findings_text}\n"
            f"\nIMPRESSION:\n"
            f"{impression}\n"
            f"\nRECOMMENDATION:\n"
            f"{recommendation}\n"
            f"\n[AI-Assisted Report — Requires Radiologist Review]\n"
        )

        return {
            "report_text": report_text,
            "findings": findings_text,
            "impression": impression,
            "recommendation": recommendation,
            "disease_name": disease_name,
            "severity": severity_str,
            "affected_levels": affected_levels,
            "confidence": confidence,
            "pfirrmann_grade": pfirrmann,
        }


# ─────────────────────────────────────────────────────────────────────
# Neural Report Generator (optional, for learned generation)
# ─────────────────────────────────────────────────────────────────────

class NeuralReportHead(nn.Module):
    """
    Lightweight neural report encoder.
    Maps fused features to a report embedding for retrieval-based generation.
    Not a full autoregressive decoder — uses template augmentation + retrieval.
    """

    def __init__(
        self,
        input_dim: int = 512,
        report_embed_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, report_embed_dim * 2),
            nn.LayerNorm(report_embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(report_embed_dim * 2, report_embed_dim),
            nn.LayerNorm(report_embed_dim),
        )
        # Predict key report attributes
        self.disease_pred = nn.Linear(report_embed_dim, 7)
        self.severity_pred = nn.Linear(report_embed_dim, 3)
        self.level_pred = nn.Linear(report_embed_dim, 8)
        self.pfirrmann_pred = nn.Sequential(
            nn.Linear(report_embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, fused_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        emb = self.encoder(fused_features)
        return {
            "report_embedding": emb,
            "disease_logits": self.disease_pred(emb),
            "severity_logits": self.severity_pred(emb),
            "level_logits": self.level_pred(emb),
            "pfirrmann_score": self.pfirrmann_pred(emb).squeeze(-1) * 4 + 1,
        }


def format_predictions_for_report(model_output: Dict) -> Dict:
    """
    Convert raw model output tensors to Python scalars/lists for report generation.
    """
    result = {}

    # Disease
    if "disease" in model_output:
        dis = model_output["disease"]
        result["disease_pred"] = int(dis["pred"][0]) if dis["pred"].numel() > 0 else 0
        result["disease_confidence"] = float(dis["confidence"][0]) if dis["confidence"].numel() > 0 else 0.0
    else:
        result["disease_pred"] = 0
        result["disease_confidence"] = 0.0

    # Severity
    if "severity" in model_output:
        sev = model_output["severity"]
        result["severity_pred"] = int(sev["pred"][0]) if sev["pred"].numel() > 0 else 0
    else:
        result["severity_pred"] = 0

    # Level
    if "level" in model_output:
        lvl = model_output["level"]
        result["level_pred"] = lvl["pred"][0].cpu().tolist() if lvl["pred"].numel() > 0 else [0] * 8
    else:
        result["level_pred"] = [0] * 8

    # Pfirrmann
    if "ivd_pathology" in model_output:
        ivd = model_output["ivd_pathology"]
        result["pfirrmann_score"] = float(ivd["pfirrmann_score"][0]) if ivd["pfirrmann_score"].numel() > 0 else 3.0
    else:
        result["pfirrmann_score"] = 3.0

    return result
