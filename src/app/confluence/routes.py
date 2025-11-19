"""Webhook routes for Confluence page ingestion."""
from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

from ..config import Settings, get_settings
from ..embeddings.ingestion import PageIngestionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["webhook"])


def get_ingestion_service(settings: Settings = Depends(get_settings)) -> PageIngestionService:
    return PageIngestionService(settings)


@router.post("/confluence")
async def ingest_confluence_page(
    request: Request,
    background: BackgroundTasks,
    ingestion_service: PageIngestionService = Depends(get_ingestion_service),
) -> Dict[str, Any]:
    payload = await request.json()
    page_id = payload.get("pageId") or payload.get("page_id")
    if not page_id:
        logger.warning("Webhook payload missing pageId: %s", payload)
        raise HTTPException(status_code=400, detail="Missing pageId in payload")

    background.add_task(ingestion_service.process_page, str(page_id))
    return {"status": "accepted", "page_id": page_id}
