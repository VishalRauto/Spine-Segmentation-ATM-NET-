"""
SQLAlchemy ORM models for ATM-Net++ database.

Tables:
- users
- patients
- studies
- predictions
- reports
- audit_logs
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, JSON, String, Text, Enum as SAEnum
)
from sqlalchemy.types import TypeDecorator, CHAR
import sqlalchemy.dialects.postgresql as pg_types
from sqlalchemy.orm import DeclarativeBase, relationship
import enum


# ── UUID type that works with both PostgreSQL and SQLite ──────────────
class GUID(TypeDecorator):
    """Platform-independent GUID type. Uses PostgreSQL UUID natively,
    stores as CHAR(36) string on other backends (e.g. SQLite)."""
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(pg_types.UUID())
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        import uuid as _uuid
        return _uuid.UUID(str(value))


def utcnow():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────

class UserRole(str, enum.Enum):
    ADMIN = "admin"
    RADIOLOGIST = "radiologist"
    CLINICIAN = "clinician"
    RESEARCHER = "researcher"
    VIEWER = "viewer"


class StudyStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REVIEWED = "reviewed"


class DiseaseType(str, enum.Enum):
    NORMAL = "Normal"
    DISC_HERNIATION = "Disc_Herniation"
    DISC_BULGE = "Disc_Bulge"
    SPINAL_STENOSIS = "Spinal_Stenosis"
    DDD = "Degenerative_Disc_Disease"
    SPONDYLOLISTHESIS = "Spondylolisthesis"
    COMPRESSION_FRACTURE = "Compression_Fracture"


class SeverityLevel(str, enum.Enum):
    MILD = "Mild"
    MODERATE = "Moderate"
    SEVERE = "Severe"


# ─────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(SAEnum(UserRole), default=UserRole.CLINICIAN, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    last_login = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    patients = relationship("Patient", back_populates="created_by_user", foreign_keys="Patient.created_by")
    studies = relationship("Study", back_populates="created_by_user", foreign_keys="Study.created_by")


class Patient(Base):
    __tablename__ = "patients"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    patient_code = Column(String(50), unique=True, nullable=False, index=True)
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    date_of_birth = Column(DateTime, nullable=True)
    sex = Column(String(10), nullable=True)
    age = Column(Integer, nullable=True)
    height_cm = Column(Float, nullable=True)
    weight_kg = Column(Float, nullable=True)
    bmi = Column(Float, nullable=True)
    clinical_symptoms = Column(Text, nullable=True)
    medical_history = Column(JSON, nullable=True)
    created_by = Column(GUID(), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    # Relationships
    created_by_user = relationship("User", back_populates="patients", foreign_keys=[created_by])
    studies = relationship("Study", back_populates="patient")


class Study(Base):
    __tablename__ = "studies"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    study_uid = Column(String(100), unique=True, nullable=False, index=True)
    patient_id = Column(GUID(), ForeignKey("patients.id"), nullable=False)
    created_by = Column(GUID(), ForeignKey("users.id"), nullable=True)

    # MRI metadata
    modality = Column(String(10), default="T2")
    image_path = Column(String(500), nullable=True)
    image_filename = Column(String(255), nullable=True)
    image_size_bytes = Column(Integer, nullable=True)
    manufacturer = Column(String(100), nullable=True)
    magnetic_field_strength = Column(Float, nullable=True)
    pixel_spacing = Column(JSON, nullable=True)
    slice_thickness = Column(Float, nullable=True)
    num_slices = Column(Integer, nullable=True)

    # Report text
    radiology_report = Column(Text, nullable=True)

    # Status
    status = Column(SAEnum(StudyStatus), default=StudyStatus.PENDING)
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    processed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    patient = relationship("Patient", back_populates="studies")
    created_by_user = relationship("User", back_populates="studies", foreign_keys=[created_by])
    predictions = relationship("Prediction", back_populates="study")
    reports = relationship("Report", back_populates="study")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    study_id = Column(GUID(), ForeignKey("studies.id"), nullable=False)

    # Model info
    model_version = Column(String(50), default="1.0.0")

    # Segmentation results
    seg_mask_path = Column(String(500), nullable=True)
    seg_overlay_path = Column(String(500), nullable=True)
    mean_dice = Column(Float, nullable=True)
    detected_structures = Column(JSON, nullable=True)

    # Disease classification
    disease_id = Column(Integer, nullable=True)
    disease_name = Column(SAEnum(DiseaseType), nullable=True)
    disease_confidence = Column(Float, nullable=True)
    disease_probabilities = Column(JSON, nullable=True)

    # Severity
    severity = Column(SAEnum(SeverityLevel), nullable=True)
    severity_score = Column(Float, nullable=True)

    # Affected levels
    affected_levels = Column(JSON, nullable=True)
    level_probabilities = Column(JSON, nullable=True)

    # Pfirrmann
    pfirrmann_grade = Column(Float, nullable=True)

    # Explainability
    gradcam_path = Column(String(500), nullable=True)

    # Inference metadata
    inference_time_ms = Column(Float, nullable=True)
    num_slices_processed = Column(Integer, nullable=True)
    raw_output = Column(JSON, nullable=True)

    created_at = Column(DateTime(timezone=True), default=utcnow)

    # Relationships
    study = relationship("Study", back_populates="predictions")


class Report(Base):
    __tablename__ = "reports"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    study_id = Column(GUID(), ForeignKey("studies.id"), nullable=False)
    prediction_id = Column(GUID(), ForeignKey("predictions.id"), nullable=True)

    report_text = Column(Text, nullable=False)
    findings = Column(Text, nullable=True)
    impression = Column(Text, nullable=True)
    recommendation = Column(Text, nullable=True)
    pdf_path = Column(String(500), nullable=True)
    is_reviewed = Column(Boolean, default=False)
    reviewed_by = Column(GUID(), ForeignKey("users.id"), nullable=True)
    review_notes = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    # Relationships
    study = relationship("Study", back_populates="reports")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=True)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(100), nullable=True)
    resource_id = Column(String(100), nullable=True)
    details = Column(JSON, nullable=True)
    ip_address = Column(String(50), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
