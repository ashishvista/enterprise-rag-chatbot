"""FastAPI routes for testing retriever results."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from .service import RetrievalResult, RetrieverService
from llama_index.core.schema import NodeWithScore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/retriever", tags=["retriever"])


_retriever_cache: Optional[Tuple[str, RetrieverService]] = None


def get_retriever_service(settings: Settings = Depends(get_settings)) -> RetrieverService:
    global _retriever_cache
    signature = json.dumps(settings.model_dump(mode="json"), sort_keys=True)
    if _retriever_cache is None or _retriever_cache[0] != signature:
        _retriever_cache = (signature, RetrieverService(settings))
    return _retriever_cache[1]


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query to search for.")
    top_k: Optional[int] = Field(
        None,
        ge=1,
        description="Number of top results to return after reranking.",
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


def _serialize_node(node_with_score: NodeWithScore) -> RetrievedNode:
    node = node_with_score.node
    node_id = getattr(node, "node_id", None) or getattr(node, "id_", None) or getattr(node, "doc_id", None)
    try:
        text = node.get_content()  # type: ignore[attr-defined]
    except AttributeError:
        text = getattr(node, "text", "") or ""
    metadata = getattr(node, "metadata", None)
    if metadata is None:
        metadata_dict: Dict[str, Any] | None = None
    elif isinstance(metadata, dict):
        metadata_dict = metadata
    else:
        metadata_dict = dict(metadata)
    return RetrievedNode(
        node_id=str(node_id or ""),
        score=float(node_with_score.score),
        text=text,
        metadata=metadata_dict,
    )


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
        result: RetrievalResult = service.retrieve(payload.query, payload.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Retriever query failed")
        raise HTTPException(status_code=500, detail="Retriever query failed") from exc

    hits = [_serialize_node(node) for node in result.reranked_nodes]
    return RetrieveResponse(
        top_k=desired_top_k,
        total_hits=len(result.raw_hits),
        results=hits,
    )
