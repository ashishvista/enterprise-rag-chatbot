"""Chatbot FastAPI routes combining RAG, LangChain, and conversation persistence."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from ..retriever.dependencies import get_retriever_service
from ..retriever.service import RetrieverService
from .service import ChatResult, ChatbotService
from llama_index.core.schema import NodeWithScore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chatbot", tags=["chatbot"])


_chatbot_cache: Optional[tuple[str, ChatbotService]] = None


def get_chatbot_service(
    settings: Settings = Depends(get_settings),
    retriever: RetrieverService = Depends(get_retriever_service),
) -> ChatbotService:
    global _chatbot_cache
    signature = json.dumps(settings.model_dump(mode="json"), sort_keys=True)
    cache_key = f"{signature}|{id(retriever)}"
    if _chatbot_cache is None or _chatbot_cache[0] != cache_key:
        _chatbot_cache = (cache_key, ChatbotService(settings, retriever))
    return _chatbot_cache[1]


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="Opaque session identifier.")
    message: str = Field(..., min_length=1, description="User utterance to answer.")
    top_k: Optional[int] = Field(
        None,
        ge=1,
        description="Optional override for top results to rerank from the retriever.",
    )


class SourceDocument(BaseModel):
    node_id: str
    score: float
    text: str
    metadata: Dict[str, Any] | None = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    context: str
    sources: List[SourceDocument]


def _serialize_sources(nodes: Sequence[NodeWithScore]) -> List[SourceDocument]:
    serialized: List[SourceDocument] = []
    for node_with_score in nodes:
        node = node_with_score.node
        node_id = getattr(node, "node_id", None) or getattr(node, "id_", None) or getattr(node, "doc_id", None)
        try:
            text = node.get_content()  # type: ignore[attr-defined]
        except AttributeError:
            text = getattr(node, "text", "")
        metadata = getattr(node, "metadata", None)
        if metadata is None:
            metadata_dict: Dict[str, Any] | None = None
        elif isinstance(metadata, dict):
            metadata_dict = metadata
        else:
            metadata_dict = dict(metadata)
        serialized.append(
            SourceDocument(
                node_id=str(node_id or ""),
                score=float(node_with_score.score),
                text=text or "",
                metadata=metadata_dict,
            )
        )
    return serialized


@router.post("/respond", response_model=ChatResponse)
async def respond(
    payload: ChatRequest, service: ChatbotService = Depends(get_chatbot_service)
) -> ChatResponse:
    if payload.top_k and payload.top_k > service.settings.retriever_search_k:
        raise HTTPException(status_code=400, detail="top_k cannot exceed RETRIEVER_SEARCH_K")
    try:
        result: ChatResult = await service.generate_response(
            session_id=payload.session_id,
            user_message=payload.message,
            top_k=payload.top_k,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("Chatbot response generation failed")
        raise HTTPException(status_code=500, detail="Chatbot response generation failed") from exc

    sources = _serialize_sources(list(result.sources))
    return ChatResponse(
        session_id=result.session_id,
        response=result.response,
        context=result.context,
        sources=sources,
    )
