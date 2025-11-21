"""Shared state definitions for the chatbot workflow."""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from llama_index.core.schema import NodeWithScore


class ChatState(TypedDict, total=False):
    session_id: str
    user_message: str
    top_k: Optional[int]
    history_messages: Sequence[BaseMessage]
    sources: List[NodeWithScore]
    raw_hits: List[NodeWithScore]
    context: str
    response: str
    observer: Any
