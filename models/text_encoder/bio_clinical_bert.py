"""
Bio-ClinicalBERT Text Encoder for ATM-Net++.

Encodes radiology reports into dense embeddings using
emilyalsentzer/Bio_ClinicalBERT (or BioBERT as fallback).

Also includes:
- Named entity extraction for spine-specific pathology
- Level detection from text
- Severity extraction
"""

from __future__ import annotations

import re
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Spine anatomy keywords for rule-based fallback
SPINE_LEVELS = ["T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5", "S1",
                "L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1",
                "T12/L1", "T11/T12", "T10/T11"]

PATHOLOGY_KEYWORDS = {
    "disc_herniation": ["herniation", "herniated", "protrusion", "extruded disc"],
    "disc_bulge": ["bulge", "bulging", "broad-based"],
    "spinal_stenosis": ["stenosis", "canal narrowing", "foraminal narrowing", "central stenosis"],
    "degenerative": ["degeneration", "degenerative", "desiccation", "dessication",
                     "height loss", "pfirrmann", "osteophyte"],
    "spondylolisthesis": ["spondylolisthesis", "retrolisthesis", "anterolisthesis"],
    "fracture": ["fracture", "compression fracture", "endplate fracture", "wedge"],
    "modic": ["modic", "marrow edema", "end plate signal"],
}

SEVERITY_KEYWORDS = {
    "mild": ["mild", "minimal", "slight", "minor"],
    "moderate": ["moderate", "moderate-sized", "significant"],
    "severe": ["severe", "marked", "critical", "complete", "extreme"],
}


class ClinicalTextEncoder(nn.Module):
    """
    Encodes radiology report text using Bio-ClinicalBERT.
    Produces:
    - CLS token embedding (global representation)
    - Token embeddings (for cross-attention with image)
    - Projected feature vector for fusion
    """

    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        max_length: int = 512,
        embedding_dim: int = 768,
        output_dim: int = 768,
        freeze_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.freeze_layers = freeze_layers

        # Load BERT model and tokenizer
        self._bert = None
        self._tokenizer = None
        self._load_bert()

        # Projection head: BERT embedding -> fusion dim
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Pathology classifier heads (auxiliary)
        self.pathology_heads = nn.ModuleDict({
            name: nn.Linear(output_dim, 1)
            for name in PATHOLOGY_KEYWORDS.keys()
        })
        self.severity_head = nn.Linear(output_dim, 3)

    def _load_bert(self):
        """Load Bio-ClinicalBERT with graceful fallback to bert-base-uncased."""
        try:
            from transformers import AutoTokenizer, AutoModel
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._bert = AutoModel.from_pretrained(self.model_name)
            logger.info(f"Loaded {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to load {self.model_name}: {e}. Falling back to bert-base-uncased")
            try:
                from transformers import AutoTokenizer, AutoModel
                self._tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                self._bert = AutoModel.from_pretrained("bert-base-uncased")
            except Exception as e2:
                logger.error(f"Failed to load fallback BERT: {e2}")
                self._bert = None
                self._tokenizer = None

        if self._bert is not None:
            # Freeze early layers
            for i, layer in enumerate(self._bert.encoder.layer):
                if i < self.freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

    @property
    def tokenizer(self):
        return self._tokenizer

    def tokenize(
        self,
        texts: List[str],
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize a list of report strings.

        Returns:
            dict with 'input_ids', 'attention_mask', 'token_type_ids'
        """
        if self._tokenizer is None:
            # Return dummy tokens
            bs = len(texts)
            seq_len = 64
            return {
                "input_ids": torch.zeros(bs, seq_len, dtype=torch.long),
                "attention_mask": torch.ones(bs, seq_len, dtype=torch.long),
            }

        encoding = self._tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        if device is not None:
            encoding = {k: v.to(device) for k, v in encoding.items()}
        return encoding

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        return_token_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (B, seq_len)
            attention_mask: (B, seq_len)
            token_type_ids: (B, seq_len), optional
            return_token_embeddings: Also return per-token embeddings

        Returns:
            dict with:
                'cls_embedding': (B, output_dim) - global report feature
                'token_embeddings': (B, seq_len, embedding_dim) - if requested
                'pathology_logits': dict of (B, 1) per pathology
                'severity_logits': (B, 3)
        """
        if self._bert is None:
            B = input_ids.shape[0]
            dummy_feat = torch.zeros(B, self.output_dim, device=input_ids.device)
            return {"cls_embedding": dummy_feat, "severity_logits": torch.zeros(B, 3, device=input_ids.device)}

        bert_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            bert_kwargs["token_type_ids"] = token_type_ids

        outputs = self._bert(**bert_kwargs)

        # CLS token: (B, embedding_dim)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        # Sequence output: (B, seq_len, embedding_dim)
        token_emb = outputs.last_hidden_state

        # Project
        projected = self.projection(cls_emb)

        result = {
            "cls_embedding": projected,
            "severity_logits": self.severity_head(projected),
        }

        # Per-pathology predictions
        pathology_logits = {}
        for name, head in self.pathology_heads.items():
            pathology_logits[name] = head(projected)
        result["pathology_logits"] = pathology_logits

        if return_token_embeddings:
            result["token_embeddings"] = token_emb

        return result

    def encode_text(self, texts: List[str], device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Convenience method: tokenize and encode.
        Returns (B, output_dim) embedding tensor.
        """
        if device is None:
            device = next(self.parameters()).device

        tokens = self.tokenize(texts, device=device)
        with torch.no_grad():
            out = self.forward(**tokens)
        return out["cls_embedding"]


class RuleBasedTextParser:
    """
    Rule-based fallback parser for radiology report text.
    Extracts pathology, levels, and severity when BERT is not available.
    """

    def parse(self, report_text: str) -> Dict:
        text_lower = report_text.lower()
        result = {
            "pathologies": [],
            "levels": [],
            "severity": "mild",
            "raw_text": report_text,
        }

        # Extract pathologies
        for pathology, keywords in PATHOLOGY_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in text_lower:
                    result["pathologies"].append(pathology)
                    break

        # Extract levels
        for level in SPINE_LEVELS:
            pattern = level.replace("/", r"[/\-]")
            if re.search(pattern, report_text, re.IGNORECASE):
                if level not in result["levels"]:
                    result["levels"].append(level)

        # Extract severity
        for severity, keywords in SEVERITY_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in text_lower:
                    result["severity"] = severity
                    break

        return result

    def to_feature_vector(self, parsed: Dict, dim: int = 768) -> torch.Tensor:
        """
        Convert parsed report to a simple feature vector.
        Used as fallback when BERT is unavailable.
        """
        feature = torch.zeros(dim)

        # Binary pathology features (first 7 dims)
        path_list = ["disc_herniation", "disc_bulge", "spinal_stenosis",
                     "degenerative", "spondylolisthesis", "fracture", "modic"]
        for i, p in enumerate(path_list):
            if p in parsed["pathologies"]:
                feature[i] = 1.0

        # Severity (dims 7-9)
        severity_map = {"mild": 0, "moderate": 1, "severe": 2}
        sev_idx = 7 + severity_map.get(parsed["severity"], 0)
        feature[sev_idx] = 1.0

        # Level features (dims 10-20)
        level_map = {
            "L1": 10, "L2": 11, "L3": 12, "L4": 13, "L5": 14, "S1": 15,
            "L4/L5": 16, "L5/S1": 17, "L3/L4": 18, "L2/L3": 19, "L1/L2": 20,
        }
        for level, idx in level_map.items():
            if level in parsed["levels"] and idx < dim:
                feature[idx] = 1.0

        return feature.unsqueeze(0)  # (1, dim)
