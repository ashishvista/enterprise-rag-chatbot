"""NatWest knowledge base tool that calls the retriever HTTP API."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..config import get_settings
from ..config.http_client import create_async_httpx_client


class KnowledgeBaseInput(BaseModel):
    """Schema defining NatWest knowledge base lookup parameters."""

    query: str = Field(
        ...,
        description="Natural language question to search the NatWest knowledge base with.",
    )
    labels: Optional[List[str]] = Field(
        default=None,
        description="Optional labels to filter retrieval; leave empty to search all.",
    )


@tool(
    "natwest_knowledge_base",
    description=(
        "Access the NatWest enterprise knowledge base covering technology, finance, careers, employee programs, "
        "HR, payroll, and related internal topics. Required argument: query (string)."
    ),
    args_schema=KnowledgeBaseInput,
)
async def query_natwest_knowledge_base(query: str, labels: Optional[List[str]] = None) -> str:
    """Return a formatted answer using the /retriever/query HTTP endpoint."""

    question = (query or "").strip()
    if not question:
        return "Please provide a question about NatWest to search the knowledge base."

    settings = get_settings()
    base_url = (
        getattr(settings, "service_base_url", None)
        or os.getenv("SERVICE_BASE_URL")
        or "http://127.0.0.1:8000"
    )

    try:
        async with create_async_httpx_client(base_url=base_url, timeout=settings.request_timeout) as client:
            response = await client.post(
                "/retriever/query",
                json={"query": question, "labels": labels},
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 503:
            return "The NatWest knowledge base is currently unavailable."
        return "Failed to query the NatWest knowledge base due to an internal error."
    except httpx.HTTPError:
        return "Failed to query the NatWest knowledge base due to an internal error."

    payload = response.json()
    results = payload.get("results") or []
    return results
