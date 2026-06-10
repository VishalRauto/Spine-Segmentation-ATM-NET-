"""Report routes: generate, download PDF, list."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.api.middleware.auth_middleware import get_current_active_user
from backend.api.schemas.schemas import ReportResponse
from backend.db.database import get_db
from backend.db.models.models import Report, Study, Prediction, User

router = APIRouter(prefix="/reports", tags=["Reports"])


@router.get("/{report_id}", response_model=ReportResponse)
async def get_report(
    report_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Report).where(Report.id == report_id))
    report = result.scalar_one_or_none()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


@router.get("/study/{study_id}", response_model=ReportResponse)
async def get_report_by_study(
    study_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Report).where(Report.study_id == study_id).order_by(Report.created_at.desc())
    )
    report = result.scalars().first()
    if not report:
        raise HTTPException(status_code=404, detail="No report found for study")
    return report


@router.post("/generate/{study_id}")
async def generate_report(
    study_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate and save a report for a completed study."""
    study_result = await db.execute(select(Study).where(Study.id == study_id))
    study = study_result.scalar_one_or_none()
    if not study:
        raise HTTPException(status_code=404, detail="Study not found")

    # Get latest prediction
    pred_result = await db.execute(
        select(Prediction).where(Prediction.study_id == study_id).order_by(Prediction.created_at.desc())
    )
    prediction = pred_result.scalars().first()
    if not prediction:
        raise HTTPException(status_code=404, detail="No prediction found. Run prediction first.")

    from models.report_generator.clinical_report import TemplateReportGenerator, format_predictions_for_report

    pred_dict = {
        "disease_pred": prediction.disease_id or 0,
        "disease_confidence": prediction.disease_confidence or 0.0,
        "severity_pred": ["Mild", "Moderate", "Severe"].index(prediction.severity.value)
                         if prediction.severity else 0,
        "level_pred": [1 if l in (prediction.affected_levels or []) else 0
                       for l in ["T10/T11","T11/T12","T12/L1","L1/L2","L2/L3","L3/L4","L4/L5","L5/S1"]],
        "pfirrmann_score": prediction.pfirrmann_grade or 3.0,
    }
    gen = TemplateReportGenerator()
    report_data = gen.generate(pred_dict)

    report = Report(
        study_id=study_id,
        prediction_id=prediction.id,
        report_text=report_data["report_text"],
        findings=report_data["findings"],
        impression=report_data["impression"],
        recommendation=report_data["recommendation"],
    )
    db.add(report)
    await db.flush()
    await db.refresh(report)
    return {"report_id": str(report.id), "message": "Report generated successfully"}


@router.get("/download/{report_id}/pdf")
async def download_pdf(
    report_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Download a report as PDF."""
    result = await db.execute(select(Report).where(Report.id == report_id))
    report = result.scalar_one_or_none()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    from backend.services.pdf_service import generate_pdf_report

    prediction_mock = {
        "report": {
            "report_text": report.report_text,
            "findings": report.findings or "",
            "impression": report.impression or "",
            "recommendation": report.recommendation or "",
            "disease_name": "N/A",
            "severity": "N/A",
            "affected_levels": [],
            "confidence": 0.0,
            "pfirrmann_grade": 0.0,
        },
        "classification": {"disease_name": "N/A", "confidence": 0.0, "disease_probabilities": {}},
        "severity": {"name": "N/A"},
        "levels": {"affected": []},
        "pfirrmann_grade": 0.0,
    }

    pdf_bytes = generate_pdf_report(prediction_mock)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="report_{report_id}.pdf"'},
    )
