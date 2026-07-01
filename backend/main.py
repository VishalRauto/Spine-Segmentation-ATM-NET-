"""
ATM-Net++ FastAPI Application Entry Point.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.config import get_settings
from backend.db.database import create_tables

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown lifecycle."""
    logger.info("Starting ATM-Net++ API server...")

    # Create DB tables
    try:
        await create_tables()
        logger.info("Database tables ready")
    except Exception as e:
        logger.warning(f"DB setup: {e}")

    # Warm up model
    try:
        from backend.services.model_service import get_predictor
        await get_predictor()
        logger.info("Model loaded and ready")
    except Exception as e:
        logger.warning(f"Model warm-up: {e}")

    # Pre-warm NLP models in background (Bio-ClinicalBERT + zero-shot)
    import asyncio as _asyncio
    async def _prewarm_nlp():
        try:
            from server import get_bert, get_zero_shot
            await _asyncio.get_event_loop().run_in_executor(None, get_bert)
            logger.info("Bio-ClinicalBERT pre-warmed")
            await _asyncio.get_event_loop().run_in_executor(None, get_zero_shot)
            logger.info("Zero-shot classifier pre-warmed")
        except Exception as _e:
            logger.info(f"NLP pre-warm skipped: {_e}")
    _asyncio.ensure_future(_prewarm_nlp())

    yield

    logger.info("Shutting down ATM-Net++ API server")


# ── App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "ATM-Net++: Anatomy-Aware Multimodal Lumbar Spine MRI Diagnostic API. "
        "Provides segmentation, disease classification, severity estimation, "
        "and clinical report generation."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.middleware("http")
async def add_process_time(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{(time.time()-t0)*1000:.2f}ms"
    return response


# ── Routes ────────────────────────────────────────────────────────────

from backend.api.routes import auth, predict, patients, reports, analytics

app.include_router(auth.router, prefix="/api/v1")
app.include_router(predict.router, prefix="/api/v1")
app.include_router(patients.router, prefix="/api/v1")
app.include_router(reports.router, prefix="/api/v1")
app.include_router(analytics.router, prefix="/api/v1")


# ── Static files (uploaded images, outputs) ──────────────────────────

Path("uploads").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


# ── Health check ─────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
async def health():
    import torch
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "cuda_available": torch.cuda.is_available(),
        "device": settings.MODEL_DEVICE,
    }


@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "ATM-Net++ Spine AI API",
        "docs": "/docs",
        "version": settings.APP_VERSION,
    }


# ── Exception handlers ────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )


if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS if not settings.DEBUG else 1,
        log_level="info",
    )
