"""Dependency helpers for embeddings ingestion."""
from __future__ import annotations

from fastapi import Depends

from ..config import Settings, get_settings
from .ingestion import PageIngestionService


def get_ingestion_service(
    settings: Settings = Depends(get_settings),
) -> PageIngestionService:
    return PageIngestionService(settings)
