"""Typed state definition for the LangGraph-powered agent."""
from __future__ import annotations

from typing import Any, Dict, List

from typing_extensions import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


def append_invocations(
    existing: List[Dict[str, Any]] | None,
    new: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Append tool invocation summaries to the running list."""
    combined = list(existing or [])
    combined.extend(new)
    return combined


class AgentState(TypedDict, total=False):
    """Canonical state shared across LangGraph nodes."""

    messages: Annotated[List[AnyMessage], add_messages]
    pending_tool_calls: List[Dict[str, Any]]
    tool_invocations: Annotated[List[Dict[str, Any]], append_invocations]
