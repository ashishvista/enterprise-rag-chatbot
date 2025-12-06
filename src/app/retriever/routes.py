"""FastAPI routes for testing retriever results."""
from __future__ import annotations

from dataclasses import asdict
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from .dependencies import get_retriever_service
from .service import RetrievalResult, RetrieverService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/retriever", tags=["retriever"])


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query to search for.")
    top_k: Optional[int] = Field(
        None,
        ge=1,
        description="Number of top results to return after reranking.",
    )
    labels: Optional[List[str]] = Field(
        default=None,
        description="Optional list of labels to filter on (OR across provided labels).",
    )


class RetrievedNode(BaseModel):
    node_id: str
    score: float
    text: str
    metadata: Dict[str, Any] | None = None


class RetrieveResponse(BaseModel):
    top_k: int
    total_hits: int
    results: List[RetrievedNode]


@router.post("/query", response_model=RetrieveResponse)
async def query_retriever(
    payload: RetrieveRequest, service: RetrieverService = Depends(get_retriever_service)
) -> RetrieveResponse:
    desired_top_k = payload.top_k or service.settings.retriever_top_k
    if desired_top_k > service.settings.retriever_search_k:
        raise HTTPException(
            status_code=400,
            detail="top_k cannot exceed RETRIEVER_SEARCH_K",
        )
    try:
        result: RetrievalResult = await service.retrieve(payload.query, payload.top_k, labels=payload.labels)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Retriever query failed")
        raise HTTPException(status_code=500, detail="Retriever query failed") from exc

    serialized_hits = [service.serialize_node(node) for node in result.reranked_nodes]
    hits = [RetrievedNode(**asdict(item)) for item in serialized_hits]
    return RetrieveResponse(
        top_k=desired_top_k,
        total_hits=len(result.raw_hits),
        results=hits,
    )
