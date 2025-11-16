"""FastAPI router that handles Confluence webhooks."""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .page_pipeline import PageIngestionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["webhooks"])

ACCEPTED_EVENTS = {"page_created", "page_updated"}


class ConfluenceContentInfo(BaseModel):
    """Subset of the Confluence content payload we care about."""

    id: str = Field(..., alias="id")
    type: str
    title: Optional[str] = None


class ConfluenceWebhookEvent(BaseModel):
    """Primary webhook payload body."""

    event_type: Optional[str] = Field(None, alias="eventType")
    webhook_event: Optional[str] = Field(None, alias="webhookEvent")
    content: ConfluenceContentInfo

    def resolved_event(self) -> Optional[str]:
        return self.event_type or self.webhook_event


@router.post("/confluence", status_code=202)
def handle_confluence_webhook(
    payload: ConfluenceWebhookEvent,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
) -> dict:
    """Accept and enqueue Confluence page events."""
    event_name = payload.resolved_event()
    if event_name not in ACCEPTED_EVENTS:
        logger.info("Ignoring unsupported Confluence webhook event: %s", event_name)
        return {"status": "ignored", "reason": "unsupported event"}
    page_id = payload.content.id
    service = PageIngestionService(settings)
    background_tasks.add_task(service.process_page, page_id)
    logger.info("Enqueued Confluence page %s for ingestion", page_id)
    return {"status": "accepted", "event": event_name, "page_id": page_id}
