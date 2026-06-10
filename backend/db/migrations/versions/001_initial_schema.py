"""Initial schema — all ATM-Net++ tables

Revision ID: 001_initial
Revises:
Create Date: 2024-01-01 00:00:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # users
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("username", sa.String(100), nullable=False, unique=True),
        sa.Column("full_name", sa.String(255), nullable=True),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("role", sa.String(50), nullable=False, server_default="clinician"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("is_verified", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("last_login", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_users_email", "users", ["email"])
    op.create_index("ix_users_username", "users", ["username"])

    # patients
    op.create_table(
        "patients",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("patient_code", sa.String(50), nullable=False, unique=True),
        sa.Column("first_name", sa.String(100), nullable=True),
        sa.Column("last_name", sa.String(100), nullable=True),
        sa.Column("date_of_birth", sa.DateTime, nullable=True),
        sa.Column("sex", sa.String(10), nullable=True),
        sa.Column("age", sa.Integer, nullable=True),
        sa.Column("height_cm", sa.Float, nullable=True),
        sa.Column("weight_kg", sa.Float, nullable=True),
        sa.Column("bmi", sa.Float, nullable=True),
        sa.Column("clinical_symptoms", sa.Text, nullable=True),
        sa.Column("medical_history", postgresql.JSON, nullable=True),
        sa.Column("created_by", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("users.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_patients_code", "patients", ["patient_code"])

    # studies
    op.create_table(
        "studies",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("study_uid", sa.String(100), nullable=False, unique=True),
        sa.Column("patient_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("patients.id"), nullable=False),
        sa.Column("created_by", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("users.id"), nullable=True),
        sa.Column("modality", sa.String(10), server_default="T2"),
        sa.Column("image_path", sa.String(500), nullable=True),
        sa.Column("image_filename", sa.String(255), nullable=True),
        sa.Column("image_size_bytes", sa.Integer, nullable=True),
        sa.Column("manufacturer", sa.String(100), nullable=True),
        sa.Column("magnetic_field_strength", sa.Float, nullable=True),
        sa.Column("pixel_spacing", postgresql.JSON, nullable=True),
        sa.Column("slice_thickness", sa.Float, nullable=True),
        sa.Column("num_slices", sa.Integer, nullable=True),
        sa.Column("radiology_report", sa.Text, nullable=True),
        sa.Column("status", sa.String(20), server_default="pending"),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_studies_uid", "studies", ["study_uid"])
    op.create_index("ix_studies_patient", "studies", ["patient_id"])

    # predictions
    op.create_table(
        "predictions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("study_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("studies.id"), nullable=False),
        sa.Column("model_version", sa.String(50), server_default="1.0.0"),
        sa.Column("seg_mask_path", sa.String(500), nullable=True),
        sa.Column("seg_overlay_path", sa.String(500), nullable=True),
        sa.Column("mean_dice", sa.Float, nullable=True),
        sa.Column("detected_structures", postgresql.JSON, nullable=True),
        sa.Column("disease_id", sa.Integer, nullable=True),
        sa.Column("disease_name", sa.String(50), nullable=True),
        sa.Column("disease_confidence", sa.Float, nullable=True),
        sa.Column("disease_probabilities", postgresql.JSON, nullable=True),
        sa.Column("severity", sa.String(20), nullable=True),
        sa.Column("severity_score", sa.Float, nullable=True),
        sa.Column("affected_levels", postgresql.JSON, nullable=True),
        sa.Column("level_probabilities", postgresql.JSON, nullable=True),
        sa.Column("pfirrmann_grade", sa.Float, nullable=True),
        sa.Column("gradcam_path", sa.String(500), nullable=True),
        sa.Column("inference_time_ms", sa.Float, nullable=True),
        sa.Column("num_slices_processed", sa.Integer, nullable=True),
        sa.Column("raw_output", postgresql.JSON, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_predictions_study", "predictions", ["study_id"])

    # reports
    op.create_table(
        "reports",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("study_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("studies.id"), nullable=False),
        sa.Column("prediction_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("predictions.id"), nullable=True),
        sa.Column("report_text", sa.Text, nullable=False),
        sa.Column("findings", sa.Text, nullable=True),
        sa.Column("impression", sa.Text, nullable=True),
        sa.Column("recommendation", sa.Text, nullable=True),
        sa.Column("pdf_path", sa.String(500), nullable=True),
        sa.Column("is_reviewed", sa.Boolean, server_default="false"),
        sa.Column("reviewed_by", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("users.id"), nullable=True),
        sa.Column("review_notes", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # audit_logs
    op.create_table(
        "audit_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("users.id"), nullable=True),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("resource_type", sa.String(100), nullable=True),
        sa.Column("resource_id", sa.String(100), nullable=True),
        sa.Column("details", postgresql.JSON, nullable=True),
        sa.Column("ip_address", sa.String(50), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("audit_logs")
    op.drop_table("reports")
    op.drop_table("predictions")
    op.drop_table("studies")
    op.drop_table("patients")
    op.drop_table("users")
