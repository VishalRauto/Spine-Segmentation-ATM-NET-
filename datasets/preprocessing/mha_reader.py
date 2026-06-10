"""
MHA/MHD/NIfTI/DICOM reader with full metadata extraction.
Supports the SPIDER dataset format (MHA files with vertebrae + disc labels).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MedicalImageMeta:
    """Container for medical image metadata."""
    path: str
    spacing: Tuple[float, ...] = field(default_factory=lambda: (1.0, 1.0, 1.0))
    origin: Tuple[float, ...] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    direction: Optional[np.ndarray] = None
    shape: Tuple[int, ...] = field(default_factory=tuple)
    dtype: str = "float32"
    modality: str = "unknown"
    patient_id: Optional[str] = None
    series_description: Optional[str] = None
    magnetic_field_strength: Optional[float] = None
    manufacturer: Optional[str] = None
    pixel_spacing: Optional[List[float]] = None
    slice_thickness: Optional[float] = None


@dataclass
class MedicalImage:
    """Medical image with pixel data and metadata."""
    data: np.ndarray
    meta: MedicalImageMeta

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def spacing(self) -> Tuple[float, ...]:
        return self.meta.spacing

    def as_float32(self) -> "MedicalImage":
        return MedicalImage(
            data=self.data.astype(np.float32),
            meta=self.meta
        )


class MedicalImageReader:
    """
    Unified reader for MHA, MHD, NIfTI, DICOM formats.
    Handles the SPIDER dataset (SimpleITK MHA files).
    """

    SUPPORTED_EXTENSIONS = {".mha", ".mhd", ".nii", ".nii.gz", ".dcm"}

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._sitk = None
        self._nibabel = None
        self._pydicom = None

    def _get_sitk(self):
        if self._sitk is None:
            import SimpleITK as sitk  # noqa: F401
            self._sitk = sitk
        return self._sitk

    def _get_nibabel(self):
        if self._nibabel is None:
            import nibabel as nib  # noqa: F401
            self._nibabel = nib
        return self._nibabel

    def _get_pydicom(self):
        if self._pydicom is None:
            import pydicom  # noqa: F401
            self._pydicom = pydicom
        return self._pydicom

    def read(self, path: Union[str, Path]) -> MedicalImage:
        """Read a medical image from any supported format."""
        path = Path(path)
        suffix = "".join(path.suffixes).lower()

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        if suffix in {".mha", ".mhd"}:
            return self._read_mha(path)
        elif suffix in {".nii", ".nii.gz"}:
            return self._read_nifti(path)
        elif suffix == ".dcm" or path.is_dir():
            return self._read_dicom(path)
        else:
            # Try SimpleITK as fallback
            return self._read_mha(path)

    def _read_mha(self, path: Path) -> MedicalImage:
        """Read MHA/MHD file using SimpleITK."""
        sitk = self._get_sitk()
        img = sitk.ReadImage(str(path))

        data = sitk.GetArrayFromImage(img)  # shape: (Z, Y, X) or (Y, X)
        spacing = img.GetSpacing()           # (X, Y, Z)
        origin = img.GetOrigin()
        direction = np.array(img.GetDirection())

        # Reorder spacing to match array order (Z, Y, X)
        if len(spacing) == 3:
            spacing_ordered = (spacing[2], spacing[1], spacing[0])
        else:
            spacing_ordered = spacing

        meta = MedicalImageMeta(
            path=str(path),
            spacing=spacing_ordered,
            origin=origin,
            direction=direction,
            shape=data.shape,
            dtype=str(data.dtype),
            patient_id=path.stem,
        )

        # Detect modality from filename
        stem = path.stem.lower()
        if "_t1" in stem:
            meta.modality = "T1"
        elif "_t2" in stem:
            meta.modality = "T2"

        if self.verbose:
            logger.info(f"Read MHA: {path.name}, shape={data.shape}, spacing={spacing}")

        return MedicalImage(data=data, meta=meta)

    def _read_nifti(self, path: Path) -> MedicalImage:
        """Read NIfTI file using nibabel."""
        nib = self._get_nibabel()
        img = nib.load(str(path))
        data = np.asarray(img.dataobj)  # (X, Y, Z) in NIfTI convention
        data = np.transpose(data, (2, 1, 0))  # Convert to (Z, Y, X)

        header = img.header
        zooms = header.get_zooms()
        spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0])) if len(zooms) >= 3 else zooms

        meta = MedicalImageMeta(
            path=str(path),
            spacing=spacing,
            shape=data.shape,
            dtype=str(data.dtype),
            patient_id=path.stem.replace(".nii", ""),
        )
        return MedicalImage(data=data, meta=meta)

    def _read_dicom(self, path: Path) -> MedicalImage:
        """Read DICOM series from directory or single file."""
        sitk = self._get_sitk()

        if path.is_dir():
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(str(path))
            if not dicom_names:
                raise ValueError(f"No DICOM series found in {path}")
            reader.SetFileNames(dicom_names)
            img = reader.Execute()
        else:
            img = sitk.ReadImage(str(path))

        data = sitk.GetArrayFromImage(img)
        spacing = img.GetSpacing()

        meta = MedicalImageMeta(
            path=str(path),
            spacing=(spacing[2], spacing[1], spacing[0]) if len(spacing) == 3 else spacing,
            origin=img.GetOrigin(),
            shape=data.shape,
            dtype=str(data.dtype),
            modality="MRI",
        )
        return MedicalImage(data=data, meta=meta)

    def read_spider_pair(
        self, patient_id: str, images_dir: str, masks_dir: str, modality: str = "t2"
    ) -> Tuple[MedicalImage, MedicalImage]:
        """
        Read image + mask pair from the SPIDER dataset.

        Args:
            patient_id: e.g., "100"
            images_dir: path to images folder
            masks_dir: path to masks folder
            modality: "t1" or "t2"

        Returns:
            (image, mask) tuple
        """
        fname = f"{patient_id}_{modality}.mha"
        img_path = Path(images_dir) / fname
        mask_path = Path(masks_dir) / fname

        image = self.read(img_path)
        mask = self.read(mask_path)

        return image, mask

    def save(self, image: MedicalImage, path: Union[str, Path]) -> None:
        """Save a MedicalImage back to disk."""
        sitk = self._get_sitk()
        path = Path(path)

        img_sitk = sitk.GetImageFromArray(image.data)
        if image.meta.spacing:
            # Convert (Z,Y,X) spacing back to (X,Y,Z)
            sp = image.meta.spacing
            img_sitk.SetSpacing(tuple(reversed(sp)) if len(sp) == 3 else sp)
        if image.meta.origin:
            img_sitk.SetOrigin(image.meta.origin)

        sitk.WriteImage(img_sitk, str(path))
        logger.info(f"Saved image to {path}")
