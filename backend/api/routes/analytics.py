"""Analytics dashboard routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, select

from backend.api.middleware.auth_middleware import get_current_active_user
from backend.api.schemas.schemas import AnalyticsSummary
from backend.db.database import get_db
from backend.db.models.models import Patient, Prediction, Study, User

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/summary", response_model=AnalyticsSummary)
async def get_summary(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    total_studies = (await db.execute(select(func.count()).select_from(Study))).scalar()
    total_patients = (await db.execute(select(func.count()).select_from(Patient))).scalar()
    total_predictions = (await db.execute(select(func.count()).select_from(Prediction))).scalar()

    # Disease distribution
    dis_result = await db.execute(
        select(Prediction.disease_name, func.count().label("cnt"))
        .group_by(Prediction.disease_name)
    )
    disease_dist = {str(r[0]): r[1] for r in dis_result.all() if r[0]}

    # Severity distribution
    sev_result = await db.execute(
        select(Prediction.severity, func.count().label("cnt"))
        .group_by(Prediction.severity)
    )
    severity_dist = {str(r[0]): r[1] for r in sev_result.all() if r[0]}

    # Average inference time
    avg_time = (
        await db.execute(select(func.avg(Prediction.inference_time_ms)))
    ).scalar()

    # Dice score average
    avg_dice = (
        await db.execute(select(func.avg(Prediction.mean_dice)))
    ).scalar()

    return AnalyticsSummary(
        total_studies=total_studies or 0,
        total_patients=total_patients or 0,
        total_predictions=total_predictions or 0,
        disease_distribution=disease_dist,
        severity_distribution=severity_dist,
        average_dice=round(float(avg_dice), 4) if avg_dice else None,
        average_inference_time_ms=round(float(avg_time), 2) if avg_time else None,
    )


@router.get("/model-performance")
async def get_model_performance(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Return model performance metrics summary."""
    result = await db.execute(
        select(
            func.avg(Prediction.mean_dice).label("avg_dice"),
            func.avg(Prediction.disease_confidence).label("avg_confidence"),
            func.avg(Prediction.inference_time_ms).label("avg_time"),
            func.count().label("total"),
        )
    )
    row = result.one()
    return {
        "average_dice": round(float(row.avg_dice), 4) if row.avg_dice else None,
        "average_confidence": round(float(row.avg_confidence), 4) if row.avg_confidence else None,
        "average_inference_ms": round(float(row.avg_time), 2) if row.avg_time else None,
        "total_predictions": row.total,
    }
