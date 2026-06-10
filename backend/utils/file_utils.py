"""
File handling utilities for the ATM-Net++ backend.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".mha", ".mhd", ".nii", ".gz", ".dcm", ".png", ".jpg", ".jpeg"}
MAX_FILE_SIZE_MB = 500


def validate_upload_file(filename: str, content: bytes) -> tuple[bool, str]:
    """
    Validate an uploaded file.
    Returns (is_valid, error_message).
    """
    # Check extension
    ext = "".join(Path(filename).suffixes).lower()
    if ext not in ALLOWED_EXTENSIONS and not filename.lower().endswith(".nii.gz"):
        return False, f"Unsupported format: {ext}. Allowed: {ALLOWED_EXTENSIONS}"

    # Check size
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"File too large: {size_mb:.1f}MB. Max: {MAX_FILE_SIZE_MB}MB"

    # Basic magic bytes check
    if ext == ".dcm" and not content[:4] == b"DICM" and content[128:132] != b"DICM":
        # DICOM files may or may not have the preamble
        pass  # Allow — some DICOM files lack the magic bytes

    return True, ""


def save_upload(
    content: bytes,
    filename: str,
    upload_dir: str,
    prefix: Optional[str] = None,
) -> str:
    """
    Save uploaded file to disk with a UUID-prefixed name.
    Returns the full path.
    """
    upload_path = Path(upload_dir)
    upload_path.mkdir(parents=True, exist_ok=True)

    ext = "".join(Path(filename).suffixes).lower()
    unique_name = f"{prefix or ''}{uuid.uuid4()}{ext}"
    full_path = upload_path / unique_name

    with open(full_path, "wb") as f:
        f.write(content)

    logger.debug(f"Saved upload: {full_path} ({len(content)/1024:.1f} KB)")
    return str(full_path)


def compute_file_hash(path: str, algorithm: str = "md5") -> str:
    """Compute MD5/SHA256 hash of a file for deduplication."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def cleanup_old_uploads(upload_dir: str, max_age_hours: int = 24):
    """Remove upload files older than max_age_hours."""
    import time
    now = time.time()
    cutoff = now - max_age_hours * 3600
    deleted = 0
    for p in Path(upload_dir).iterdir():
        if p.is_file() and p.stat().st_mtime < cutoff:
            p.unlink()
            deleted += 1
    if deleted:
        logger.info(f"Cleaned up {deleted} old upload files from {upload_dir}")


def list_checkpoints(checkpoint_dir: str) -> List[dict]:
    """List all checkpoints with metadata."""
    import torch
    results = []
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return results

    for p in sorted(ckpt_dir.glob("*.pth")):
        try:
            ckpt = torch.load(p, map_location="cpu")
            results.append({
                "filename": p.name,
                "path": str(p),
                "epoch": ckpt.get("epoch", "?"),
                "best_dice": ckpt.get("best_dice", 0.0),
                "size_mb": round(p.stat().st_size / 1e6, 1),
            })
        except Exception:
            results.append({"filename": p.name, "path": str(p), "error": "Could not load"})

    return results
