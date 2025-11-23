"""LangGraph workflow definition for the chatbot service."""
from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Dict, Optional, Sequence

from langgraph.graph import END, START, StateGraph
from langchain_core.tools import BaseTool

from .state import ChatState

_MERMAID_PATH_ENV = "CHATBOT_WORKFLOW_MERMAID_PATH"

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from .service import ChatbotService


def build_chat_workflow(service: "ChatbotService", tools: Sequence[BaseTool]):
    """Return a compiled LangGraph workflow bound to the provided service."""

    graph = StateGraph(ChatState)
    tool_map = {tool.name: tool for tool in tools}

    def _resolve_diagram_path(settings: "ChatbotService".settings.__class__) -> Optional[str]:
        if getattr(settings, "chatbot_workflow_mermaid_path", None):
            return settings.chatbot_workflow_mermaid_path
        env_path = os.getenv(_MERMAID_PATH_ENV)
        if env_path:
            return env_path
        return None

    def _write_mermaid_png(compiled_graph, settings: "ChatbotService".settings.__class__) -> None:
        diagram_path = _resolve_diagram_path(settings)
        if not diagram_path:
            return

        from pathlib import Path

        try:
            path = Path(diagram_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            png_bytes = compiled_graph.get_graph().draw_mermaid_png()
            path.write_bytes(png_bytes)
        except Exception:  # pragma: no cover - visualization is best effort
            pass

    async def retrieve_context(state: ChatState) -> ChatState:
        top_k = state.get("top_k")
        retrieval = await service._retriever.retrieve(state["user_message"], top_k)
        sources = list(retrieval.reranked_nodes or [])
        raw_hits = list(retrieval.raw_hits)

        context = service._format_context(sources)
        result: ChatState = {
            "sources": sources,
            "raw_hits": raw_hits,
            "context": context,
        }
        observer = state.get("observer")
        if observer is not None:
            before_snapshot = dict(state)
            after_snapshot = dict(before_snapshot)
            after_snapshot.update(result)
            await observer.record_node("retrieve_context", before_snapshot, after_snapshot)
        return result

    async def run_llm(state: ChatState) -> ChatState:
        prompt_inputs: Dict[str, object] = {
            "system_prompt": service.settings.chat_system_prompt,
            # "context": state.get("context") or "No enterprise context retrieved.",
            "context": state.get("context") or "",

            "history": state.get("history_messages", []),
            "question": state["user_message"],
            "tool_instructions": service.tool_instructions,
        }
        response_text = await service._invoke_chain(prompt_inputs)
        response_text = response_text.strip()
        result = {"response": response_text}
        observer = state.get("observer")
        if observer is not None:
            before_snapshot = dict(state)
            after_snapshot = dict(before_snapshot)
            after_snapshot.update(result)
            await observer.record_node("run_llm", before_snapshot, after_snapshot)
        return result

    async def parse_tool(state: ChatState) -> ChatState:
        response_text = state.get("response") or ""
        tool_request = None
        if response_text:
            try:
                parsed = json.loads(response_text)
                if isinstance(parsed, dict) and "tool" in parsed:
                    tool_request = parsed
            except json.JSONDecodeError:
                tool_request = None

        tool_name = tool_request.get("tool") if tool_request else None
        result: ChatState = {
            "tool_request": tool_request,
            "tool_result": None,
            "tool_name": tool_name,
        }
        observer = state.get("observer")
        if observer is not None:
            before_snapshot = dict(state)
            after_snapshot = dict(before_snapshot)
            after_snapshot.update(result)
            await observer.record_node("parse_tool", before_snapshot, after_snapshot)
        return result

    async def invoke_tool(state: ChatState) -> ChatState:
        request = state.get("tool_request") or {}
        tool_name = request.get("tool")
        args = request.get("args") or {}
        tool = tool_map.get(tool_name)
        if tool is None:
            tool_output = f"Requested tool '{tool_name}' is unavailable."
        else:
            try:
                tool_output = await tool.ainvoke(args)
            except NotImplementedError:
                tool_output = tool.invoke(args)
            except Exception as exc:  # pragma: no cover - tool execution guard
                tool_output = f"Tool '{tool_name}' failed: {exc}"
        if not isinstance(tool_output, str):
            tool_output = str(tool_output)
        result: ChatState = {
            "tool_result": tool_output,
            "tool_name": tool_name,
        }
        observer = state.get("observer")
        if observer is not None:
            before_snapshot = dict(state)
            after_snapshot = dict(before_snapshot)
            after_snapshot.update(result)
            await observer.record_node("invoke_tool", before_snapshot, after_snapshot)
        return result

    async def compose_tool_response(state: ChatState) -> ChatState:
        tool_request = state.get("tool_request") or {}
        tool_name = tool_request.get("tool")
        tool_result = state.get("tool_result")
        context = state.get("context") or "No enterprise context retrieved."
        tool_context = ""
        if tool_name and tool_result:
            tool_context = f"\n\nTool {tool_name} output:\n{tool_result}"
        prompt_inputs: Dict[str, object] = {
            "system_prompt": service.settings.chat_system_prompt,
            "context": context + tool_context,
            "history": state.get("history_messages", []),
            "question": state["user_message"],
            "tool_instructions": service.tool_instructions,
        }
        response_text = await service._invoke_chain(prompt_inputs)
        response_text = response_text.strip()
        result: ChatState = {
            "response": response_text,
            # "tool_request": None,
            # "tool_result": tool_result,
            # "tool_name": tool_name,
        }
        observer = state.get("observer")
        if observer is not None:
            before_snapshot = dict(state)
            after_snapshot = dict(before_snapshot)
            after_snapshot.update(result)
            await observer.record_node("compose_tool_response", before_snapshot, after_snapshot)
        return result

    async def store_response(state: ChatState) -> ChatState:
        response_text = state.get("response") or ""
        await service._history_store.add_message(state["session_id"], "assistant", response_text)
        result = {"response": response_text}
        observer = state.get("observer")
        if observer is not None:
            before_snapshot = dict(state)
            after_snapshot = dict(before_snapshot)
            after_snapshot.update(result)
            await observer.record_node("store_response", before_snapshot, after_snapshot)
        return result

    # graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("run_llm", run_llm)
    graph.add_node("parse_tool", parse_tool)
    graph.add_node("invoke_tool", invoke_tool)
    graph.add_node("compose_tool_response", compose_tool_response)
    graph.add_node("store_response", store_response)

    def _tool_decision(state: ChatState) -> str:
        return "invoke" if state.get("tool_request") else "store"

    graph.add_edge(START, "retrieve_context")
    graph.add_edge("retrieve_context", "run_llm")
    graph.add_edge("run_llm", "parse_tool")
    graph.add_conditional_edges(
        "parse_tool",
        _tool_decision,
        {
            "invoke": "invoke_tool",
            "store": "store_response",
        },
    )
    graph.add_edge("invoke_tool", "compose_tool_response")
    graph.add_edge("compose_tool_response", "parse_tool")

    graph.add_conditional_edges(
        "parse_tool",
        _tool_decision,
        {
            "invoke": "invoke_tool",
            "store": "store_response",
        },
    )
    # graph.add_edge("compose_tool_response", "store_response")
    graph.add_edge("store_response", END)

    compiled_graph = graph.compile()

    _write_mermaid_png(compiled_graph, service.settings)

    return compiled_graph
