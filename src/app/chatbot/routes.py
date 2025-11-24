"""FastAPI route exposing the LangGraph-based chatbot."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Sequence

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from ..conversation_history.store import ConversationHistoryStore, ConversationMessage
from ..llm.ollama_chat import create_chat_model
from ..observability.langfuse import create_langfuse_observer
from ..tools import get_default_tools
from .graph import LangGraphAgent

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

_agent_cache: tuple[str, LangGraphAgent] | None = None


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="Opaque session identifier.")
    message: str = Field(..., min_length=1, description="User utterance for the assistant.")


class ChatMessageModel(BaseModel):
    role: str
    content: str


class ToolCallModel(BaseModel):
    id: str = ""
    name: str = ""
    arguments: str = ""
    result: str | None = None
    error: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    messages: List[ChatMessageModel]
    tool_calls: List[ToolCallModel]


def _agent_signature(settings: Settings) -> str:
    payload = {
        "ollama_base_url": settings.ollama_base_url,
        "model": settings.llm_model_name,
        "temperature": settings.llm_temperature,
        "max_output": settings.llm_max_output_tokens,
        "context_window": settings.llm_context_window,
    }
    return json.dumps(payload, sort_keys=True)


def get_agent(settings: Settings = Depends(get_settings)) -> LangGraphAgent:
    global _agent_cache
    signature = _agent_signature(settings)
    if _agent_cache is None or _agent_cache[0] != signature:
        llm = create_chat_model(settings)
        tools = list(get_default_tools())
        _agent_cache = (signature, LangGraphAgent(llm, tools))
    return _agent_cache[1]


def _history_to_messages(records: Sequence[ConversationMessage]) -> List[BaseMessage]:
    messages: List[BaseMessage] = []
    for record in records:
        if record.role == "user":
            messages.append(HumanMessage(content=record.content))
        elif record.role == "assistant":
            messages.append(AIMessage(content=record.content))
        elif record.role == "tool":
            tool_id = f"history-{record.message_index}"
            messages.append(
                ToolMessage(
                    content=record.content,
                    name="history_tool",
                    tool_call_id=tool_id,
                )
            )
        else:
            messages.append(HumanMessage(content=record.content))
    return messages


def _message_role(message: BaseMessage) -> str:
    mapping = {"human": "user", "ai": "assistant", "tool": "tool"}
    return mapping.get(message.type, message.type)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if value is None:
                    value = json.dumps(item, ensure_ascii=True)
                parts.append(str(value))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    try:
        return json.dumps(content, ensure_ascii=True)
    except TypeError:
        return str(content)


def _message_to_text(message: BaseMessage) -> str:
    return _content_to_text(message.content)


def _serialize_messages(messages: Sequence[BaseMessage]) -> List[ChatMessageModel]:
    return [
        ChatMessageModel(role=_message_role(message), content=_message_to_text(message))
        for message in messages
    ]


def _merge_tool_metadata(
    declared_calls: Sequence[Dict[str, Any]] | None,
    invocations: Sequence[Dict[str, Any]] | None,
) -> List[Dict[str, Any]]:
    declared_calls = declared_calls or []
    invocations = invocations or []
    invocation_map = {entry.get("id", ""): entry for entry in invocations}
    merged: List[Dict[str, Any]] = []
    for call in declared_calls:
        call_id = call.get("id", "")
        entry = {
            "id": call_id,
            "name": call.get("name", ""),
            "arguments": call.get("arguments", ""),
            "result": None,
            "error": None,
        }
        details = invocation_map.get(call_id)
        if details:
            entry["result"] = details.get("result")
            entry["error"] = details.get("error")
        merged.append(entry)
    for call_id, details in invocation_map.items():
        if not any(item["id"] == call_id for item in merged):
            merged.append(
                {
                    "id": call_id,
                    "name": details.get("name", ""),
                    "arguments": details.get("arguments", ""),
                    "result": details.get("result"),
                    "error": details.get("error"),
                }
            )
    return merged


def _non_system_messages(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    return [message for message in messages if message.type != "system"]


async def _persist_new_messages(
    store: ConversationHistoryStore,
    session_id: str,
    previous_count: int,
    messages: Sequence[BaseMessage],
) -> None:
    non_system = _non_system_messages(messages)
    new_segment = non_system[previous_count:]
    payload: List[tuple[str, str]] = []
    for message in new_segment:
        role = _message_role(message)
        payload.append((role, _message_to_text(message)))
    if payload:
        await store.add_messages(session_id, payload)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    agent: LangGraphAgent = Depends(get_agent),
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    store = ConversationHistoryStore(settings)
    history = await store.fetch_recent_messages(
        session_id=payload.session_id,
        limit=settings.conversation_history_max_messages,
    )
    history_messages = _history_to_messages(history)
    conversation: List[BaseMessage] = [
        SystemMessage(content=settings.chat_system_prompt),
        *history_messages,
        HumanMessage(content=payload.message),
    ]
    observer = create_langfuse_observer(
        settings,
        session_id=payload.session_id,
        user_message=payload.message,
    )
    try:
        final_state = await agent.run(conversation, observer=observer)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("Chat agent execution failed for session %s", payload.session_id)
        raise HTTPException(status_code=500, detail="Chat agent execution failed") from exc

    agent_messages = final_state.get("messages", [])
    if not agent_messages:
        raise HTTPException(status_code=500, detail="Agent produced no messages")

    await _persist_new_messages(
        store=store,
        session_id=payload.session_id,
        previous_count=len(history),
        messages=agent_messages,
    )

    non_system = _non_system_messages(agent_messages)
    reply = next((msg for msg in reversed(non_system) if msg.type == "ai"), None)
    if reply is None:
        raise HTTPException(status_code=500, detail="Agent produced no assistant reply")

    tool_calls = _merge_tool_metadata(
        final_state.get("tool_calls"),
        final_state.get("tool_invocations"),
    )

    return ChatResponse(
        session_id=payload.session_id,
        reply=_message_to_text(reply),
        messages=_serialize_messages(non_system),
        tool_calls=[ToolCallModel(**item) for item in tool_calls],
    )
