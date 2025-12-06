"""Webhook routes for Confluence page ingestion."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request

import httpx

from ..config import Settings, get_settings
from ..config.http_client import create_async_httpx_client
from ..embeddings.dependencies import get_ingestion_service
from ..embeddings.ingestion import PageIngestionService
from ..embeddings.routes import EmbeddingIngestRequest
from .client import ConfluenceClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["webhook"])


@router.post("/confluence")
async def ingest_confluence_page(
    request: Request,
    ingestion_service: PageIngestionService = Depends(get_ingestion_service),
    settings: Settings = Depends(get_settings),
) -> Dict[str, Any]:
    payload = await request.json()
    page_id = payload.get("pageId") or payload.get("page_id")
    if not page_id:
        logger.warning("Webhook payload missing pageId: %s", payload)
        raise HTTPException(status_code=400, detail="Missing pageId in payload")

    await _trigger_embedding_ingest(str(page_id), ingestion_service, settings)
    return {"status": "accepted", "page_id": page_id}


@router.post("/confluence/bulk")
async def ingest_confluence_pages_bulk(
    request: Request,
    ingestion_service: PageIngestionService = Depends(get_ingestion_service),
    settings: Settings = Depends(get_settings),
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
        await _trigger_embedding_ingest(page_id_str, ingestion_service, settings)

    if not accepted:
        raise HTTPException(status_code=400, detail="No valid pageIds provided")

    return {"status": "accepted", "page_ids": accepted, "requested": len(page_ids)}


async def _trigger_embedding_ingest(
    page_id: str,
    ingestion_service: PageIngestionService,
    settings: Settings,
) -> None:
    with ConfluenceClient(settings) as client:
        page_payload = client.fetch_page(page_id)
    metadata = ConfluenceClient.page_metadata(page_payload)
    document_text = page_payload.get("body", {}).get("storage", {}).get("value", "")

    embed_request = EmbeddingIngestRequest(
        node_id=page_id,
        text=document_text,
        metadata=metadata,
        labels=metadata.get("labels"),
        document_type="confluence",
    )
    async with create_async_httpx_client(
        base_url=settings.embeddings_base_url,
        timeout=settings.request_timeout,
    ) as client:
        try:
            response = await client.post(
                "/embeddings/create",
                json=embed_request.model_dump(),
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error(
                "Failed to create embeddings for page %s via embeddings API: %s",
                page_id,
                exc,
                exc_info=True,
            )
            raise HTTPException(status_code=502, detail="Failed to create embeddings")
