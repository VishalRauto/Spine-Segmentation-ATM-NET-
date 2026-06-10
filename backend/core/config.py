"""
Backend configuration using pydantic-settings.
Reads from environment variables and .env file.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    APP_NAME: str = "ATM-Net++ API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1

    # Security
    SECRET_KEY: str = "CHANGE-THIS-IN-PRODUCTION-USE-A-STRONG-RANDOM-KEY"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://atmnet:atmnet_pass@db:5432/atmnet_db"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20

    # Redis (for caching and task queue)
    REDIS_URL: str = "redis://redis:6379/0"

    # File storage
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE_MB: int = 500
    ALLOWED_IMAGE_EXTENSIONS: List[str] = [".mha", ".mhd", ".nii", ".gz", ".dcm", ".png", ".jpg", ".jpeg"]

    # Model
    MODEL_CHECKPOINT_PATH: str = "checkpoints/atmnet_pp_best.pth"
    MODEL_CONFIG_PATH: str = "configs/base_config.yaml"
    MODEL_DEVICE: str = "auto"  # "auto", "cuda", "cpu"
    USE_TTA: bool = False

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:3001"]

    # Celery
    CELERY_BROKER_URL: str = "redis://redis:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/2"

    # PDF
    PDF_TEMPLATE_DIR: str = "backend/templates"
    PDF_OUTPUT_DIR: str = "outputs/reports"

    @field_validator("MODEL_DEVICE")
    @classmethod
    def validate_device(cls, v: str) -> str:
        import torch
        if v == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return v

    @property
    def upload_dir_path(self):
        import pathlib
        p = pathlib.Path(self.UPLOAD_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache()
def get_settings() -> Settings:
    return Settings()
