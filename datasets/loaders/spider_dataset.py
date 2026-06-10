"""
SPIDER Dataset loader for ATM-Net++.

The SPIDER dataset (Lumbar Spine MRI) contains:
- Sagittal T1/T2 MRI images (.mha)
- Corresponding segmentation masks (.mha)
- overview.csv: patient metadata and MRI acquisition parameters
- radiological_gradings.csv: per-IVD pathology grades

Dataset structure:
  images/{patient_id}_{modality}.mha
  masks/{patient_id}_{modality}.mha

Each volume is processed slice-by-slice for 2D training.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

logger = logging.getLogger(__name__)


class SPIDERDataset(Dataset):
    """
    PyTorch Dataset for the SPIDER lumbar spine MRI dataset.

    Each __getitem__ returns a single 2D slice with its label.
    Supports:
    - T1/T2 dual-modality
    - On-the-fly preprocessing and augmentation
    - Demographic feature loading
    - Per-patient radiological grading labels
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        overview_csv: str,
        gradings_csv: str,
        patient_ids: List[str],
        modalities: List[str] = ("t1", "t2"),
        target_size: Tuple[int, int] = (512, 512),
        transform=None,
        preprocessor=None,
        label_mapper=None,
        include_demographics: bool = True,
        include_text: bool = False,
        split: str = "train",
        slice_axis: int = 0,
        min_foreground_ratio: float = 0.01,
    ):
        """
        Args:
            images_dir: Directory with image .mha files
            masks_dir: Directory with mask .mha files
            overview_csv: Path to overview.csv
            gradings_csv: Path to radiological_gradings.csv
            patient_ids: List of patient IDs to include (e.g. ["100", "101"])
            modalities: Which modalities to load (["t1"], ["t2"], or ["t1", "t2"])
            target_size: Target (H, W) for resizing
            transform: SpineAugmentor instance
            preprocessor: SpinePreprocessor instance
            label_mapper: Function to remap mask labels
            include_demographics: Include demographic features
            include_text: Include synthetic radiology report text
            split: "train", "val", or "test"
            slice_axis: Axis to slice along (0=sagittal)
            min_foreground_ratio: Minimum ratio of non-background pixels to include slice
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.patient_ids = patient_ids
        self.modalities = list(modalities)
        self.target_size = target_size
        self.transform = transform
        self.preprocessor = preprocessor
        self.label_mapper = label_mapper
        self.include_demographics = include_demographics
        self.include_text = include_text
        self.split = split
        self.slice_axis = slice_axis
        self.min_foreground_ratio = min_foreground_ratio

        # Load metadata
        self.overview_df = pd.read_csv(overview_csv)
        self.gradings_df = pd.read_csv(gradings_csv)

        # Build slice index
        self.slice_index: List[Dict] = []
        self._build_slice_index()

        logger.info(
            f"[SPIDERDataset/{split}] {len(self.slice_index)} slices from "
            f"{len(patient_ids)} patients, modalities={modalities}"
        )

    def _build_slice_index(self):
        """Pre-scan all volumes to build a flat list of (patient, modality, slice_idx) tuples."""
        from datasets.preprocessing.mha_reader import MedicalImageReader
        from datasets.preprocessing.label_mapper import remap_spider_mask

        reader = MedicalImageReader()

        for pid in self.patient_ids:
            for mod in self.modalities:
                fname = f"{pid}_{mod}.mha"
                img_path = self.images_dir / fname
                mask_path = self.masks_dir / fname

                if not img_path.exists() or not mask_path.exists():
                    logger.debug(f"Skipping missing: {fname}")
                    continue

                try:
                    img = reader.read(img_path)
                    num_slices = img.data.shape[self.slice_axis]
                except Exception as e:
                    logger.warning(f"Failed to read {img_path}: {e}")
                    continue

                for s_idx in range(num_slices):
                    self.slice_index.append({
                        "patient_id": pid,
                        "modality": mod.upper(),
                        "img_path": str(img_path),
                        "mask_path": str(mask_path),
                        "slice_idx": s_idx,
                        "total_slices": num_slices,
                    })

    def __len__(self) -> int:
        return len(self.slice_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.slice_index[idx]
        from datasets.preprocessing.mha_reader import MedicalImageReader
        from datasets.preprocessing.label_mapper import remap_spider_mask

        reader = MedicalImageReader()

        # Load image volume and extract slice
        img_vol = reader.read(entry["img_path"])
        mask_vol = reader.read(entry["mask_path"])

        s = entry["slice_idx"]
        img_slice = np.take(img_vol.data, s, axis=self.slice_axis).astype(np.float32)
        mask_slice = np.take(mask_vol.data, s, axis=self.slice_axis).astype(np.int32)

        # Remap labels
        mask_slice = remap_spider_mask(mask_slice.astype(np.int32))

        # Preprocess image
        if self.preprocessor:
            img_slice = self.preprocessor.process_slice(img_slice, modality=entry["modality"])
            if img_slice.ndim == 3:
                img_slice = img_slice[0]  # Remove channel dim for augmentor
            mask_slice = self.preprocessor.process_mask_slice(mask_slice)
        else:
            import cv2
            img_slice = cv2.resize(img_slice, (self.target_size[1], self.target_size[0]),
                                   interpolation=cv2.INTER_LINEAR)
            mask_slice = cv2.resize(mask_slice.astype(np.float32),
                                    (self.target_size[1], self.target_size[0]),
                                    interpolation=cv2.INTER_NEAREST).astype(np.int64)
            # Normalize
            p_low, p_high = np.percentile(img_slice, [0.5, 99.5])
            img_slice = np.clip(img_slice, p_low, p_high)
            img_slice = (img_slice - p_low) / (p_high - p_low + 1e-8)
            img_slice = img_slice.astype(np.float32)

        # Apply augmentation
        if self.transform and self.split == "train":
            aug_out = self.transform(img_slice, mask_slice)
            img_slice = aug_out["image"]
            mask_slice = aug_out.get("mask", mask_slice)

        # Add channel dim: (H, W) -> (1, H, W)
        if img_slice.ndim == 2:
            img_slice = img_slice[np.newaxis, ...]

        # Build output dict
        output = {
            "image": torch.from_numpy(img_slice.astype(np.float32)),
            "mask": torch.from_numpy(mask_slice.astype(np.int64)),
            "patient_id": entry["patient_id"],
            "modality": entry["modality"],
            "slice_idx": entry["slice_idx"],
        }

        # Add classification labels from gradings
        cls_labels = self._get_classification_labels(entry["patient_id"])
        output.update(cls_labels)

        # Add demographics
        if self.include_demographics:
            demo = self._get_demographics(entry["patient_id"])
            output["demographics"] = torch.tensor(demo, dtype=torch.float32)

        # Add text report embedding placeholder
        if self.include_text:
            output["report_text"] = self._get_report_text(entry["patient_id"])

        return output

    def _get_classification_labels(self, patient_id: str) -> Dict[str, torch.Tensor]:
        """
        Extract binary disease labels from radiological gradings for a patient.
        Returns aggregated patient-level labels (any-disc positive).
        """
        pid_int = int(patient_id)
        patient_grades = self.gradings_df[self.gradings_df["Patient"] == pid_int]

        if patient_grades.empty:
            return {
                "disease_label": torch.tensor(0, dtype=torch.long),
                "severity_label": torch.tensor(0, dtype=torch.long),
                "disc_herniation": torch.tensor(0.0),
                "disc_bulging": torch.tensor(0.0),
                "disc_narrowing": torch.tensor(0.0),
                "spondylolisthesis": torch.tensor(0.0),
                "pfirrmann_grade": torch.tensor(0.0),
            }

        # Aggregate across all IVDs for this patient
        disc_herniation = int(patient_grades["Disc herniation"].max() > 0) if "Disc herniation" in patient_grades.columns else 0
        disc_bulging = int(patient_grades["Disc bulging"].max() > 0) if "Disc bulging" in patient_grades.columns else 0
        disc_narrowing = int(patient_grades["Disc narrowing"].max() > 0) if "Disc narrowing" in patient_grades.columns else 0
        spondylolisthesis = int(patient_grades["Spondylolisthesis"].max() > 0) if "Spondylolisthesis" in patient_grades.columns else 0
        pfirrmann = float(patient_grades["Pfirrman grade"].mean()) if "Pfirrman grade" in patient_grades.columns else 3.0

        # Map to disease class
        if disc_herniation:
            disease = 1
        elif disc_bulging:
            disease = 2
        elif disc_narrowing:
            disease = 4  # DDD
        elif spondylolisthesis:
            disease = 5
        else:
            disease = 0  # Normal

        # Severity from Pfirrmann (1-2=mild, 3=moderate, 4-5=severe)
        if pfirrmann <= 2:
            severity = 0
        elif pfirrmann <= 3:
            severity = 1
        else:
            severity = 2

        return {
            "disease_label": torch.tensor(disease, dtype=torch.long),
            "severity_label": torch.tensor(severity, dtype=torch.long),
            "disc_herniation": torch.tensor(float(disc_herniation)),
            "disc_bulging": torch.tensor(float(disc_bulging)),
            "disc_narrowing": torch.tensor(float(disc_narrowing)),
            "spondylolisthesis": torch.tensor(float(spondylolisthesis)),
            "pfirrmann_grade": torch.tensor(pfirrmann / 5.0),
        }

    def _get_demographics(self, patient_id: str) -> np.ndarray:
        """Extract and normalize demographic features from overview.csv."""
        pid_key = f"{patient_id}_t2"
        row = self.overview_df[self.overview_df["new_file_name"] == pid_key]
        if row.empty:
            pid_key = f"{patient_id}_t1"
            row = self.overview_df[self.overview_df["new_file_name"] == pid_key]

        # Default vector: [age_norm, sex_bin, num_vertebrae_norm, num_discs_norm,
        #                  field_strength_norm, pixel_spacing_norm, te_norm, tr_norm]
        features = np.zeros(8, dtype=np.float32)

        if not row.empty:
            r = row.iloc[0]
            # Sex: F=0, M=1
            sex = str(r.get("sex", "F")).strip()
            features[0] = 1.0 if "M" in sex else 0.0
            # Normalized age (if available; dataset has birth_date)
            birth = r.get("birth_date", None)
            if pd.notna(birth):
                try:
                    features[1] = np.clip(float(birth) / 80.0, 0, 1)
                except Exception:
                    features[1] = 0.5
            # num_vertebrae normalized (typical: 5-8)
            features[2] = np.clip(float(r.get("num_vertebrae", 6)) / 8.0, 0, 1)
            features[3] = np.clip(float(r.get("num_discs", 6)) / 8.0, 0, 1)
            # MRI acquisition params (normalized)
            mfs = r.get("MagneticFieldStrength", 1.5)
            features[4] = np.clip(float(mfs) / 3.0 if pd.notna(mfs) else 0.5, 0, 1)
            try:
                ps_str = str(r.get("PixelSpacing", "[0.5, 0.5]"))
                ps = float(ps_str.strip("[]").split(",")[0])
                features[5] = np.clip(ps, 0, 1)
            except Exception:
                features[5] = 0.5
            try:
                te = float(r.get("EchoTime", 50))
                features[6] = np.clip(te / 200.0, 0, 1)
            except Exception:
                features[6] = 0.5
            try:
                tr = float(r.get("RepetitionTime", 2000))
                features[7] = np.clip(tr / 6000.0, 0, 1)
            except Exception:
                features[7] = 0.5

        return features

    def _get_report_text(self, patient_id: str) -> str:
        """Generate a synthetic radiology report text from grading data."""
        pid_int = int(patient_id)
        patient_grades = self.gradings_df[self.gradings_df["Patient"] == pid_int]

        if patient_grades.empty:
            return "No significant abnormalities identified in the lumbar spine."

        findings = []
        disc_labels = {1: "T10/T11", 2: "T11/T12", 3: "T12/L1",
                       4: "L1/L2", 5: "L2/L3", 6: "L3/L4", 7: "L4/L5"}

        for _, row in patient_grades.iterrows():
            label = int(row["IVD label"]) if "IVD label" in row else 0
            disc = disc_labels.get(label, f"disc {label}")
            pf = int(row.get("Pfirrman grade", 3))

            disc_findings = []
            if row.get("Disc herniation", 0) > 0:
                disc_findings.append("disc herniation")
            if row.get("Disc bulging", 0) > 0:
                disc_findings.append("disc bulging")
            if row.get("Disc narrowing", 0) > 0:
                disc_findings.append("disc narrowing")
            if row.get("Spondylolisthesis", 0) > 0:
                disc_findings.append("spondylolisthesis")
            if row.get("Modic", 0) > 0:
                disc_findings.append(f"Modic type {int(row['Modic'])} changes")

            if disc_findings:
                severity = "mild" if pf <= 2 else ("moderate" if pf == 3 else "severe")
                findings.append(
                    f"{severity.capitalize()} {', '.join(disc_findings)} at {disc} (Pfirrmann grade {pf})"
                )

        if not findings:
            return "Lumbar spine MRI shows no significant disc pathology. Normal alignment maintained."

        report = "Lumbar spine MRI findings: " + ". ".join(findings) + "."
        return report


def create_dataloaders(
    config: dict,
    images_dir: str,
    masks_dir: str,
    overview_csv: str,
    gradings_csv: str,
    transform_train=None,
    transform_val=None,
    preprocessor=None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Factory function to create train/val/test DataLoaders from the SPIDER dataset.

    Uses the 'subset' column in overview.csv to split patients.
    """
    overview_df = pd.read_csv(overview_csv)
    # Get unique patient IDs
    all_ids_raw = overview_df["new_file_name"].unique()

    # Extract patient IDs (strip modality suffix)
    patient_ids = set()
    for fname in all_ids_raw:
        pid = fname.replace("_t1", "").replace("_t2", "").replace("_t2_SPACE", "")
        patient_ids.add(pid)
    patient_ids = sorted(list(patient_ids))

    # Split using 'subset' column (training/validation from SPIDER)
    training_patients = set()
    validation_patients = set()
    for fname in all_ids_raw:
        row = overview_df[overview_df["new_file_name"] == fname]
        if not row.empty:
            subset = row.iloc[0].get("subset", "training")
            pid = fname.replace("_t1", "").replace("_t2", "").replace("_t2_SPACE", "")
            if subset == "training":
                training_patients.add(pid)
            else:
                validation_patients.add(pid)

    # Create a small test set from validation
    val_list = sorted(list(validation_patients))
    n_val = len(val_list)
    test_patients = set(val_list[:n_val // 2])
    val_patients = set(val_list[n_val // 2:])
    train_patients = training_patients

    modalities = config.get("data", {}).get("modalities", ["t1", "t2"])
    target_size = tuple(config.get("data", {}).get("image_size", [512, 512]))

    train_ds = SPIDERDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        overview_csv=overview_csv,
        gradings_csv=gradings_csv,
        patient_ids=sorted(list(train_patients)),
        modalities=modalities,
        target_size=target_size,
        transform=transform_train,
        preprocessor=preprocessor,
        include_demographics=True,
        include_text=True,
        split="train",
    )

    val_ds = SPIDERDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        overview_csv=overview_csv,
        gradings_csv=gradings_csv,
        patient_ids=sorted(list(val_patients)),
        modalities=modalities,
        target_size=target_size,
        transform=transform_val,
        preprocessor=preprocessor,
        include_demographics=True,
        include_text=True,
        split="val",
    )

    test_ds = SPIDERDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        overview_csv=overview_csv,
        gradings_csv=gradings_csv,
        patient_ids=sorted(list(test_patients)),
        modalities=modalities,
        target_size=target_size,
        transform=None,
        preprocessor=preprocessor,
        include_demographics=True,
        include_text=True,
        split="test",
    )

    batch_size = config.get("training", {}).get("batch_size", 4)
    num_workers = config.get("training", {}).get("num_workers", 4)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    logger.info(
        f"DataLoaders created: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)} slices"
    )
    return train_loader, val_loader, test_loader
