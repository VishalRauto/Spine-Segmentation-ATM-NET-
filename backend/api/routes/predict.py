"""
Core prediction routes:
  POST /predict/upload-mri    - Upload MRI and run prediction
  POST /predict/segment       - Segmentation only
  POST /predict/from-study    - Predict from stored study
  GET  /predict/{id}          - Fetch stored prediction
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import asyncio

from backend.api.middleware.auth_middleware import get_current_active_user
from backend.api.schemas.schemas import PredictionRequest, PredictionResponse
from backend.core.config import get_settings
from backend.db.database import get_db
from backend.db.models.models import Prediction, Study, StudyStatus, User
from backend.services.model_service import get_predictor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["Prediction"])
settings = get_settings()


def _build_demographics(
    age: Optional[int],
    sex: Optional[str],
    height: Optional[float],
    weight: Optional[float],
    bmi: Optional[float],
) -> Optional[dict]:
    if not any([age, sex, height, weight]):
        return None
    demo = {"sex": sex or "F"}
    if age: demo["age"] = age
    if height: demo["height_cm"] = height
    if weight: demo["weight_kg"] = weight
    if bmi: demo["bmi"] = bmi
    if height and weight and not bmi:
        h = height / 100
        demo["bmi"] = round(weight / (h * h), 1)
    return demo


@router.post("/upload-mri", response_model=PredictionResponse)
async def upload_and_predict(
    file: UploadFile = File(..., description="MRI file (.mha, .nii, .dcm, .png)"),
    report_text: Optional[str] = Form(None),
    modality: str = Form("T2"),
    age: Optional[int] = Form(None),
    sex: Optional[str] = Form(None),
    height_cm: Optional[float] = Form(None),
    weight_kg: Optional[float] = Form(None),
    bmi: Optional[float] = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload an MRI file and get immediate predictions."""
    # Validate file extension
    filename = file.filename or "upload"
    ext = "".join(Path(filename).suffixes).lower()
    if ext not in settings.ALLOWED_IMAGE_EXTENSIONS and ext not in {".gz", ".mha", ".nii", ".dcm", ".png", ".jpg"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {ext}. Supported: {settings.ALLOWED_IMAGE_EXTENSIONS}",
        )

    # Check file size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f}MB. Max: {settings.MAX_UPLOAD_SIZE_MB}MB",
        )

    # Save to temp file (needed only for legacy SpinePredictor path)
    upload_dir = settings.upload_dir_path
    temp_name = f"{uuid.uuid4()}{ext}"
    temp_path = upload_dir / temp_name
    with open(temp_path, "wb") as f_out:
        f_out.write(content)

    try:
        predictor = await get_predictor()
        demographics = _build_demographics(age, sex, height_cm, weight_kg, bmi)
        if demographics and report_text:
            demographics["notes"] = report_text

        # Support both ResUNetPredictor (new) and SpinePredictor (legacy)
        if hasattr(predictor, "predict"):
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: predictor.predict(content, filename, report_text, demographics)
            )
        else:
            result = predictor.predict_from_file(
                image_path=str(temp_path),
                report_text=report_text,
                demographics=demographics,
                modality=modality,
            )

        return _build_prediction_response(result)

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


@router.post("/segment", response_model=PredictionResponse)
async def segment_only(
    file: UploadFile = File(...),
    modality: str = Form("T2"),
    current_user: User = Depends(get_current_active_user),
):
    """Run segmentation only (no classification)."""
    content = await file.read()
    ext = "".join(Path(file.filename or "upload").suffixes).lower()

    upload_dir = settings.upload_dir_path
    temp_path = upload_dir / f"{uuid.uuid4()}{ext}"
    with open(temp_path, "wb") as f_out:
        f_out.write(content)

    try:
        predictor = await get_predictor()
        result = predictor.predict_from_file(str(temp_path), modality=modality)
        return _build_prediction_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


@router.post("/from-study/{study_id}", response_model=PredictionResponse)
async def predict_from_study(
    study_id: str,
    request: PredictionRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Run prediction on an already uploaded study."""
    result = await db.execute(select(Study).where(Study.study_uid == study_id))
    study = result.scalar_one_or_none()
    if not study:
        raise HTTPException(status_code=404, detail="Study not found")
    if not study.image_path or not Path(study.image_path).exists():
        raise HTTPException(status_code=404, detail="Study image file not found")

    # Update status
    study.status = StudyStatus.PROCESSING
    await db.flush()

    try:
        predictor = await get_predictor()
        demographics = None
        if request.demographics:
            demographics = request.demographics.model_dump(exclude_none=True)

        result_data = predictor.predict_from_file(
            image_path=study.image_path,
            report_text=request.report_text or study.radiology_report,
            demographics=demographics,
            modality=study.modality,
        )

        # Save prediction to DB
        pred_db = Prediction(
            study_id=study.id,
            disease_id=result_data["classification"]["disease_id"],
            disease_confidence=result_data["classification"]["confidence"],
            disease_probabilities=result_data["classification"]["disease_probabilities"],
            severity=result_data["severity"]["name"],
            pfirrmann_grade=result_data["pfirrmann_grade"],
            affected_levels=result_data["levels"]["affected"],
            inference_time_ms=result_data.get("inference_time_ms"),
            num_slices_processed=result_data.get("num_slices_processed"),
        )
        db.add(pred_db)
        study.status = StudyStatus.COMPLETED

        from datetime import datetime, timezone
        study.processed_at = datetime.now(timezone.utc)
        await db.flush()

        response = _build_prediction_response(result_data)
        response.id = pred_db.id
        response.study_id = study.id
        return response

    except Exception as e:
        study.status = StudyStatus.FAILED
        study.error_message = str(e)
        await db.flush()
        raise HTTPException(status_code=500, detail=str(e))


def _build_prediction_response(result: dict) -> PredictionResponse:
    from backend.api.schemas.schemas import (
        SegmentationResult, ClassificationResult, SeverityResult,
        LevelResult, ReportResult
    )

    report_data = result.get("report", {})

    return PredictionResponse(
        segmentation=SegmentationResult(
            overlay_b64=result.get("segmentation", {}).get("overlay_b64", ""),
            class_distribution=result.get("segmentation", {}).get("class_distribution", {}),
            detected_structures=result.get("segmentation", {}).get("detected_structures", []),
        ),
        classification=ClassificationResult(
            disease_id=result["classification"]["disease_id"],
            disease_name=result["classification"]["disease_name"],
            confidence=result["classification"]["confidence"],
            disease_probabilities=result["classification"].get("disease_probabilities", {}),
        ),
        severity=SeverityResult(
            id=result["severity"]["id"],
            name=result["severity"]["name"],
        ),
        levels=LevelResult(
            affected=result["levels"]["affected"],
            all_probs=result["levels"].get("all_probs", {}),
        ),
        pfirrmann_grade=result.get("pfirrmann_grade", 0.0),
        report=ReportResult(
            report_text=report_data.get("report_text", ""),
            findings=report_data.get("findings", ""),
            impression=report_data.get("impression", ""),
            recommendation=report_data.get("recommendation", ""),
            disease_name=report_data.get("disease_name", ""),
            severity=report_data.get("severity", ""),
            affected_levels=report_data.get("affected_levels", []),
            confidence=report_data.get("confidence", 0.0),
            pfirrmann_grade=report_data.get("pfirrmann_grade", 0.0),
        ),
        gradcam_b64=result.get("gradcam_b64", ""),
        inference_time_ms=result.get("inference_time_ms"),
        num_slices_processed=result.get("num_slices_processed"),
    )
