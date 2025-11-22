"""Webhook routes for Confluence page ingestion."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

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


@router.post("/confluence/bulk")
async def ingest_confluence_pages_bulk(
    request: Request,
    background: BackgroundTasks,
    ingestion_service: PageIngestionService = Depends(get_ingestion_service),
) -> Dict[str, Any]:
    payload = await request.json()
    page_ids: List[Any] = payload.get("pageIds") or payload.get("page_ids") or []
    if not page_ids or not isinstance(page_ids, list):
        logger.warning("Bulk webhook payload missing pageIds list: %s", payload)
        raise HTTPException(status_code=400, detail="Expected pageIds array in payload")

    accepted: List[str] = []
    for page_id in page_ids:
        if not page_id:
            continue
        page_id_str = str(page_id)
        accepted.append(page_id_str)
        background.add_task(ingestion_service.process_page, page_id_str)

    if not accepted:
        raise HTTPException(status_code=400, detail="No valid pageIds provided")

    return {"status": "accepted", "page_ids": accepted, "requested": len(page_ids)}
