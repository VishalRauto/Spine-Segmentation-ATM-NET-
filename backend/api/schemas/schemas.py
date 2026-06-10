"""
Pydantic request/response schemas for ATM-Net++ API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, field_validator


# ─────────────────────────────────────────────────────────────────────
# Auth Schemas
# ─────────────────────────────────────────────────────────────────────

class UserRegisterRequest(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    full_name: Optional[str] = None
    password: str = Field(..., min_length=8)

class UserLoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

class UserResponse(BaseModel):
    id: UUID
    email: str
    username: str
    full_name: Optional[str]
    role: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ─────────────────────────────────────────────────────────────────────
# Patient Schemas
# ─────────────────────────────────────────────────────────────────────

class PatientCreate(BaseModel):
    patient_code: str = Field(..., min_length=1, max_length=50)
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    sex: Optional[str] = Field(None, pattern="^(M|F|Other)$")
    age: Optional[int] = Field(None, ge=0, le=150)
    height_cm: Optional[float] = Field(None, ge=50, le=250)
    weight_kg: Optional[float] = Field(None, ge=1, le=500)
    bmi: Optional[float] = Field(None, ge=10, le=70)
    clinical_symptoms: Optional[str] = None

    @field_validator("bmi", mode="before")
    @classmethod
    def compute_bmi(cls, v, info):
        if v is None and info.data.get("height_cm") and info.data.get("weight_kg"):
            h = info.data["height_cm"] / 100
            return round(info.data["weight_kg"] / (h * h), 1)
        return v

class PatientResponse(BaseModel):
    id: UUID
    patient_code: str
    first_name: Optional[str]
    last_name: Optional[str]
    sex: Optional[str]
    age: Optional[int]
    height_cm: Optional[float]
    weight_kg: Optional[float]
    bmi: Optional[float]
    clinical_symptoms: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


# ─────────────────────────────────────────────────────────────────────
# Study Schemas
# ─────────────────────────────────────────────────────────────────────

class StudyCreate(BaseModel):
    patient_id: UUID
    modality: str = Field("T2", pattern="^(T1|T2|STIR|T2_SPACE)$")
    radiology_report: Optional[str] = None

class StudyResponse(BaseModel):
    id: UUID
    study_uid: str
    patient_id: UUID
    modality: str
    status: str
    image_filename: Optional[str]
    radiology_report: Optional[str]
    created_at: datetime
    processed_at: Optional[datetime]

    model_config = {"from_attributes": True}


# ─────────────────────────────────────────────────────────────────────
# Prediction Schemas
# ─────────────────────────────────────────────────────────────────────

class DemographicsInput(BaseModel):
    sex: Optional[str] = "F"
    age: Optional[int] = Field(None, ge=0, le=150)
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    bmi: Optional[float] = None
    clinical_symptoms: Optional[str] = None

class PredictionRequest(BaseModel):
    study_id: Optional[UUID] = None
    report_text: Optional[str] = None
    demographics: Optional[DemographicsInput] = None
    modality: str = "T2"
    use_tta: bool = False

class SegmentationResult(BaseModel):
    overlay_b64: Optional[str] = None
    class_distribution: Dict[str, float] = {}
    detected_structures: List[str] = []

class ClassificationResult(BaseModel):
    disease_id: int
    disease_name: str
    confidence: float
    disease_probabilities: Dict[str, float] = {}

class SeverityResult(BaseModel):
    id: int
    name: str

class LevelResult(BaseModel):
    affected: List[str] = []
    all_probs: Dict[str, float] = {}

class ReportResult(BaseModel):
    report_text: str
    findings: str
    impression: str
    recommendation: str
    disease_name: str
    severity: str
    affected_levels: List[str] = []
    confidence: float
    pfirrmann_grade: float

class PredictionResponse(BaseModel):
    id: Optional[UUID] = None
    study_id: Optional[UUID] = None
    segmentation: SegmentationResult
    classification: ClassificationResult
    severity: SeverityResult
    levels: LevelResult
    pfirrmann_grade: float
    report: ReportResult
    gradcam_b64: Optional[str] = None
    inference_time_ms: Optional[float] = None
    num_slices_processed: Optional[int] = None
    model_version: str = "1.0.0"

    model_config = {"from_attributes": True}


# ─────────────────────────────────────────────────────────────────────
# Report PDF Schemas
# ─────────────────────────────────────────────────────────────────────

class ReportResponse(BaseModel):
    id: UUID
    study_id: UUID
    report_text: str
    findings: Optional[str]
    impression: Optional[str]
    recommendation: Optional[str]
    pdf_path: Optional[str]
    is_reviewed: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ─────────────────────────────────────────────────────────────────────
# Analytics Schemas
# ─────────────────────────────────────────────────────────────────────

class AnalyticsSummary(BaseModel):
    total_studies: int
    total_patients: int
    total_predictions: int
    disease_distribution: Dict[str, int]
    severity_distribution: Dict[str, int]
    average_dice: Optional[float]
    average_inference_time_ms: Optional[float]
    studies_by_day: List[Dict[str, Any]] = []
