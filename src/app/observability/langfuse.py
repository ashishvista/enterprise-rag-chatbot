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
    if state.get("tool_request"):
        payload["tool_request"] = state["tool_request"]
    if state.get("tool_result") is not None:
        payload["tool_result"] = state["tool_result"]
    if state.get("tool_name"):
        payload["tool_name"] = state["tool_name"]
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
        init_kwargs = {"public_key": public_key, "secret_key": secret_key}
        if host:
            init_kwargs["base_url"] = host
        return _Langfuse(**init_kwargs)
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
        self._root_span: Optional[Any] = None
        self._root_span_id: Optional[str] = None
        self._initialize_trace(initial_state)

    def _initialize_trace(self, initial_state: Dict[str, Any]) -> None:
        try:
            state_payload = _serialize_state(initial_state)
            metadata = {"environment": self._environment, "session_id": self._session_id}
            root_span = self._client.start_span(
                name="chatbot_session",
                trace_context={"trace_id": self._trace_id},
                input={"state": state_payload},
                metadata=metadata,
            )
            root_span.update_trace(
                name="chatbot_session",
                user_id=self._session_id,
                session_id=self._session_id,
                input={"state": state_payload},
                metadata=metadata,
            )
            self._root_span = root_span
            self._root_span_id = root_span.id
        except Exception:  # pragma: no cover - best effort guard
            logger.exception("Failed to initialize Langfuse root span")
            self._root_span = None
            self._root_span_id = None

    async def record_node(self, name: str, before: Dict[str, Any], after: Dict[str, Any]) -> None:
        await asyncio.to_thread(self._record_node_sync, name, before, after)

    def _record_node_sync(self, name: str, before: Dict[str, Any], after: Dict[str, Any]) -> None:
        if self._root_span_id is None:
            return
        self._sequence += 1
        try:
            span = self._client.start_span(
                name=f"chatbot.{name}",
                trace_context={
                    "trace_id": self._trace_id,
                    "parent_span_id": self._root_span_id,
                },
                input={"state": _serialize_state(before)},
                output={"state": _serialize_state(after)},
                metadata={"order": self._sequence},
            )
            span.end()
        except Exception:  # pragma: no cover - best effort guard
            logger.exception("Failed to record Langfuse span for node %s", name)

    async def finalize(self, final_state: Dict[str, Any]) -> None:
        await asyncio.to_thread(self._finalize_sync, final_state)

    def _finalize_sync(self, final_state: Dict[str, Any]) -> None:
        if self._root_span is None:
            return
        try:
            serialized = _serialize_state(final_state)
            self._root_span.update(output={"state": serialized})
            self._root_span.update_trace(output={"state": serialized})
            self._root_span.end()
            self._root_span = None
            self._root_span_id = None
        except Exception:  # pragma: no cover - best effort guard
            logger.exception("Failed to finalize Langfuse trace")
        finally:
            try:
                self._client.flush()
            except Exception:  # pragma: no cover - best effort guard
                logger.exception("Failed to flush Langfuse client")


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
    trace_id = uuid.uuid4().hex
    initial_state = {"session_id": session_id, "user_message": user_message}
    return LangfuseObserver(
        client=client,
        trace_id=trace_id,
        session_id=session_id,
        environment=settings.langfuse_environment,
        initial_state=initial_state,
    )
