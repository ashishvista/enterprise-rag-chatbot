"""Langfuse observability integration helpers."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

from ..config.settings import Settings
try:  # Langfuse is optional at runtime
    from langfuse import Langfuse as _Langfuse
except ImportError:  # pragma: no cover - optional dependency
    _Langfuse = None

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langfuse import Langfuse as LangfuseType
else:
    LangfuseType = Any

logger = logging.getLogger(__name__)


def _truncate(text: str, max_len: int = 500) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _serialize_nodes(nodes: Sequence[Any]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for node_with_score in nodes:
        try:
            node = node_with_score.node
        except AttributeError:  # pragma: no cover - defensive
            node = getattr(node_with_score, "node", None)
        node_id = getattr(node, "node_id", None) or getattr(node, "id_", None) or getattr(node, "doc_id", None)
        try:
            text = node.get_content()  # type: ignore[attr-defined]
        except AttributeError:
            text = getattr(node, "text", "")
        serialized.append(
            {
                "node_id": str(node_id or ""),
                "score": float(getattr(node_with_score, "score", 0.0) or 0.0),
                "text_preview": _truncate((text or "").strip(), 280),
            }
        )
    return serialized


def _serialize_history(messages: Sequence[Any]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for message in messages:
        role = getattr(message, "type", None) or getattr(message, "role", "")
        content = getattr(message, "content", "")
        serialized.append({"role": role, "content": _truncate(str(content))})
    return serialized


def _serialize_state(state: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key in ("session_id", "user_message", "top_k", "context", "response"):
        if key in state and state[key] is not None:
            payload[key] = state[key]
    if "history_messages" in state and state["history_messages"]:
        payload["history_messages"] = _serialize_history(state["history_messages"])
    if "sources" in state and state["sources"]:
        payload["sources"] = _serialize_nodes(state["sources"])
    if "raw_hits" in state and state["raw_hits"]:
        payload["raw_hits"] = _serialize_nodes(state["raw_hits"])
    return payload


@lru_cache(maxsize=1)
def _get_langfuse_client(
    settings_signature: str,
    host: Optional[str],
    public_key: str,
    secret_key: str,
) -> Optional[LangfuseType]:
    if _Langfuse is None:
        logger.warning("Langfuse package not installed; observability is disabled")
        return None
    try:
        return _Langfuse(public_key=public_key, secret_key=secret_key, host=host or "https://cloud.langfuse.com")
    except Exception:  # pragma: no cover - best effort guard
        logger.exception("Failed to initialize Langfuse client")
        return None


class LangfuseObserver:
    """Convenience wrapper that emits LangGraph node events to Langfuse."""

    def __init__(
        self,
        *,
        client: LangfuseType,
        trace_id: str,
        session_id: str,
        environment: str,
        initial_state: Dict[str, Any],
    ) -> None:
        self._client = client
        self._trace_id = trace_id
        self._session_id = session_id
        self._environment = environment
        self._sequence = 0
        self._initialize_trace(initial_state)

    def _initialize_trace(self, initial_state: Dict[str, Any]) -> None:
        self._client.trace(
            id=self._trace_id,
            name="chatbot_session",
            user_id=self._session_id,
            metadata={
                "environment": self._environment,
                **_serialize_state(initial_state),
            },
        )

    async def record_node(self, name: str, before: Dict[str, Any], after: Dict[str, Any]) -> None:
        await asyncio.to_thread(self._record_node_sync, name, before, after)

    def _record_node_sync(self, name: str, before: Dict[str, Any], after: Dict[str, Any]) -> None:
        self._sequence += 1
        self._client.span(
            trace_id=self._trace_id,
            name=f"chatbot.{name}",
            input={"state": _serialize_state(before)},
            output={"state": _serialize_state(after)},
            metadata={"order": self._sequence},
        )

    async def finalize(self, final_state: Dict[str, Any]) -> None:
        await asyncio.to_thread(self._finalize_sync, final_state)

    def _finalize_sync(self, final_state: Dict[str, Any]) -> None:
        self._client.trace(
            id=self._trace_id,
            name="chatbot_session",
        ).update(
            output=_serialize_state(final_state),
        )
        self._client.flush()


def create_langfuse_observer(
    settings: Settings,
    *,
    session_id: str,
    user_message: str,
) -> Optional[LangfuseObserver]:
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None
    signature = json.dumps(settings.model_dump(mode="json", exclude={"langfuse_secret_key"}), sort_keys=True)
    host = settings.langfuse_host or "http://localhost:3100"
    client = _get_langfuse_client(
        settings_signature=signature,
        host=host,
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
    )
    if client is None:
        return None
    trace_id = f"chatbot-{session_id}-{uuid.uuid4()}"
    initial_state = {"session_id": session_id, "user_message": user_message}
    return LangfuseObserver(
        client=client,
        trace_id=trace_id,
        session_id=session_id,
        environment=settings.langfuse_environment,
        initial_state=initial_state,
    )
