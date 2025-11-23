"""NatWest knowledge base tool that routes queries to the retriever."""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from langchain_core.tools import tool
from llama_index.core.schema import NodeWithScore

from ..config import Settings, get_settings
from ..retriever.service import RetrieverService


@lru_cache(maxsize=1)
def _get_retriever(settings_signature: str) -> RetrieverService:
    settings = get_settings()
    return RetrieverService(settings)


def _settings_signature(settings: Settings) -> str:
    return settings.model_dump_json(sort_keys=True)


def _render_nodes(nodes: Sequence[NodeWithScore]) -> str:
    if not nodes:
        return "No relevant NatWest knowledge was found for that query."

    parts = []
    for idx, node_with_score in enumerate(nodes, start=1):
        node = node_with_score.node
        try:
            content = node.get_content()  # type: ignore[attr-defined]
        except AttributeError:
            content = getattr(node, "text", "")
        content = (content or "").strip()
        if len(content) > 500:
            content = content[:497].rstrip() + "..."
        score = getattr(node_with_score, "score", None)
        header = f"Result {idx}" if score is None else f"Result {idx} (score={score:.3f})"
        parts.append(f"{header}\n{content}")
    return "\n\n".join(parts)


@tool(
    "natwest_knowledge_base",
    description=(
        "Access the NatWest enterprise knowledge base covering finance, careers, "
        "employee programs, HR, payroll, and related internal topics. Provide a "
        "natural language question to retrieve the most relevant information."
    ),
)
async def query_natwest_knowledge_base(query: str) -> str:
    """Return a formatted answer by retrieving NatWest enterprise context."""

    question = (query or "").strip()
    if not question:
        return "Please provide a question about NatWest to search the knowledge base."

    settings = get_settings()
    retriever = _get_retriever(_settings_signature(settings))

    try:
        result = await retriever.retrieve(question)
    except RuntimeError:
        return "The NatWest knowledge base is currently unavailable."
    except Exception:  # pragma: no cover - defensive guard
        return "Failed to query the NatWest knowledge base due to an internal error."

    sources: Sequence[NodeWithScore] = result.reranked_nodes or result.raw_hits
    if not sources and result.raw_hits:
        sources = result.raw_hits

    sources = list(sources)[:3]

    return _render_nodes(sources)
