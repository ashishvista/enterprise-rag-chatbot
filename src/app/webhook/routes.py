"""FastAPI router that handles Confluence webhooks."""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from ..pipeline import PageIngestionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["webhooks"])

ACCEPTED_EVENTS = {"page_published", "page_edited"}


class ConfluenceContentInfo(BaseModel):
    """Subset of the Confluence content payload we care about."""

    id: str = Field(..., alias="id")
    type: str
    title: Optional[str] = None


class ConfluenceWebhookEvent(BaseModel):
    """Primary webhook payload body."""

    event_type: str = Field(None, alias="event_type")
    page_id: Optional[str] = None
    author: Optional[str] = None
    url: Optional[str] = None
    page_title: Optional[str] = None
    page_content: Optional[str] = None

    def resolved_event(self) -> Optional[str]:
        return self.event_type

    def resolved_page_id(self) -> Optional[str]:
        if self.page_id:
            return self.page_id
        if self.content:
            return self.content.id
        return None


@router.post("/confluence", status_code=202, response_class=JSONResponse)
async def handle_confluence_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
) -> JSONResponse:
    """Accept and enqueue Confluence page events."""
    payload_data = await request.body()
    payload = ConfluenceWebhookEvent.model_validate_json(payload_data)
    event_name = payload.resolved_event()
    if event_name not in ACCEPTED_EVENTS:
        logger.info("Ignoring unsupported Confluence webhook event: %s", event_name)
        return JSONResponse({"status": "ignored", "reason": "unsupported event"})
    page_id = payload.resolved_page_id()
    if not page_id:
        logger.warning("Received %s event without page id", event_name)
        return JSONResponse({"status": "ignored", "reason": "missing page id"})
    service = PageIngestionService(settings)
    background_tasks.add_task(service.process_page, page_id)
    logger.info("Enqueued Confluence page %s for ingestion", page_id)
    return JSONResponse({"status": "accepted", "event": event_name, "page_id": page_id})
