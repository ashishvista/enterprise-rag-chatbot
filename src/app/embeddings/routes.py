"""REST endpoints for embedding ingestion."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from .dependencies import get_ingestion_service
from .ingestion import PageIngestionService

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


class EmbeddingIngestRequest(BaseModel):
    node_id: str = Field(..., description="Unique identifier for the content being embedded")
    text: str = Field(..., description="Plaintext representation of the content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata to persist alongside embeddings")
    labels: Optional[List[str]] = Field(default=None, description="Optional Confluence labels for the content")
    document_type: Optional[str] = Field(default=None, description="Source document type, e.g., 'confluence'")


def ingest_embeddings(
    payload: EmbeddingIngestRequest,
    ingestion_service: PageIngestionService,
) -> Dict[str, Any]:
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    if payload.document_type:
        payload.metadata.setdefault("document_type", payload.document_type)

    ingestion_service.process_page(
        payload.node_id,
        document_text=payload.text,
        metadata=payload.metadata,
        labels=payload.labels,
    )
    return {"status": "accepted", "node_id": payload.node_id}


@router.post("/create")
async def create_embeddings(
    payload: EmbeddingIngestRequest,
    ingestion_service: PageIngestionService = Depends(get_ingestion_service),
) -> Dict[str, Any]:
    return ingest_embeddings(payload, ingestion_service)
