"""LangGraph application that mirrors LangChain's AgentExecutor flow."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph

from .state import AgentState

logger = logging.getLogger(__name__)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_stringify(item) for item in value]
        return "\n".join(part for part in parts if part)
    try:
        return json.dumps(value, ensure_ascii=True)
    except TypeError:
        return str(value)


def _build_tool_map(tools: Sequence[BaseTool]) -> Dict[str, BaseTool]:
    return {tool.name: tool for tool in tools}


def _normalise_tool_calls(raw_calls: Iterable[Any]) -> tuple[List[Dict[str, Any]], List[Any]]:
    cleaned: List[Dict[str, Any]] = []
    raw_list: List[Any] = []
    for call in raw_calls or []:
        raw_list.append(call)
        function = getattr(call, "function", None)
        if function is None and isinstance(call, dict):
            function = call.get("function")
        name = getattr(function, "name", None)
        if name is None and isinstance(call, dict):
            name = call.get("name")
        if name is None and isinstance(function, dict):
            name = function.get("name")
        call_id = getattr(call, "id", None)
        if call_id is None and isinstance(call, dict):
            call_id = call.get("id") or call.get("tool_call_id")
        arguments = getattr(function, "arguments", None)
        if arguments is None and isinstance(function, dict):
            arguments = function.get("arguments")
        if isinstance(arguments, bytes):
            arguments = arguments.decode("utf-8", "ignore")
        if isinstance(arguments, str):
            arguments_text = arguments
        else:
            try:
                arguments_text = json.dumps(arguments or {}, ensure_ascii=True)
            except TypeError:
                arguments_text = str(arguments)
        cleaned.append({
            "id": str(call_id or ""),
            "name": str(name or ""),
            "arguments": arguments_text,
        })
    return cleaned, raw_list


def _extract_call_details(call: Any) -> tuple[str, str, Any]:
    function = getattr(call, "function", None)
    if function is None and isinstance(call, dict):
        function = call.get("function")
    name = getattr(function, "name", None)
    if name is None and isinstance(call, dict):
        name = call.get("name")
    if name is None and isinstance(function, dict):
        name = function.get("name")
    call_id = getattr(call, "id", None)
    if call_id is None and isinstance(call, dict):
        call_id = call.get("id") or call.get("tool_call_id")
    arguments = getattr(function, "arguments", None)
    if arguments is None and isinstance(function, dict):
        arguments = function.get("arguments")
    if isinstance(arguments, bytes):
        arguments = arguments.decode("utf-8", "ignore")
    return str(call_id or ""), str(name or ""), arguments or {}


def _coerce_arguments(arguments: Any) -> tuple[Dict[str, Any], str]:
    if isinstance(arguments, str):
        text = arguments or "{}"
        try:
            parsed = json.loads(text) if text.strip() else {}
        except json.JSONDecodeError as exc:
            raise ValueError(text) from exc
        if not isinstance(parsed, dict):
            raise ValueError(text)
        return parsed, text
    if isinstance(arguments, dict):
        return arguments, json.dumps(arguments, ensure_ascii=True)
    raise ValueError(str(arguments))


def _format_result(value: Any) -> str:
    if value is None:
        return ""
    return _stringify(value)


def create_agent_app(llm: BaseChatModel, tools: Sequence[BaseTool]):
    """Compile the LangGraph workflow with LLM and tool bindings."""

    tool_map = _build_tool_map(tools)
    bound_llm = llm.bind_tools(list(tools))

    async def call_llm(state: AgentState) -> AgentState:
        conversation = list(state.get("messages", []))
        response = await bound_llm.ainvoke(conversation)
        cleaned_calls, raw_calls = _normalise_tool_calls(getattr(response, "tool_calls", []) or [])
        raw_response = _format_result(response.content)
        return {
            "messages": [response],
            "llm_input": conversation,
            "pending_tool_calls": raw_calls,
            "tool_calls": cleaned_calls,
            "raw_llm_response": raw_response,
        }

    async def call_tool(state: AgentState) -> AgentState:
        raw_queue = state.get("pending_tool_calls") or []
        if not raw_queue:
            return {"messages": []}
        messages: List[BaseMessage] = []
        invocations: List[Dict[str, Any]] = []
        for raw_call in raw_queue:
            call_id, tool_name, arguments = _extract_call_details(raw_call)
            argument_text = "{}"
            try:
                parsed_args, argument_text = _coerce_arguments(arguments)
            except ValueError as exc:
                error_text = f"Invalid arguments for tool '{tool_name}': {exc}"
                logger.warning(error_text)
                messages.append(
                    ToolMessage(
                        content=error_text,
                        name=tool_name or "unknown",
                        tool_call_id=call_id or "invalid-call",
                    )
                )
                invocations.append(
                    {
                        "id": call_id,
                        "name": tool_name,
                        "arguments": argument_text,
                        "error": error_text,
                    }
                )
                continue
            tool = tool_map.get(tool_name)
            if tool is None:
                error_text = f"Tool '{tool_name}' is not registered."
                logger.error(error_text)
                messages.append(
                    ToolMessage(
                        content=error_text,
                        name=tool_name or "unknown",
                        tool_call_id=call_id or "missing-tool",
                    )
                )
                invocations.append(
                    {
                        "id": call_id,
                        "name": tool_name,
                        "arguments": argument_text,
                        "error": error_text,
                    }
                )
                continue
            try:
                if hasattr(tool, "ainvoke"):
                    result = await tool.ainvoke(parsed_args)  # type: ignore[arg-type]
                else:
                    result = tool.invoke(parsed_args)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.exception("Tool '%s' invocation failed", tool_name)
                error_text = f"Tool '{tool_name}' raised an error: {exc}"
                messages.append(
                    ToolMessage(
                        content=error_text,
                        name=tool_name,
                        tool_call_id=call_id or "tool-error",
                    )
                )
                invocations.append(
                    {
                        "id": call_id,
                        "name": tool_name,
                        "arguments": argument_text,
                        "error": error_text,
                    }
                )
                continue
            payload = _format_result(result)
            messages.append(
                ToolMessage(
                    content=payload,
                    name=tool_name,
                    tool_call_id=call_id or "tool-result",
                )
            )
            invocations.append(
                {
                    "id": call_id,
                    "name": tool_name,
                    "arguments": argument_text,
                    "result": payload,
                }
            )
        return {
            "messages": messages,
            "pending_tool_calls": [],
            "tool_invocations": invocations,
        }

    def router(state: AgentState) -> str:
        if state.get("pending_tool_calls"):
            return "tool"
        return "end"

    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("tool", call_tool)
    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", router, {"tool": "tool", "end": END})
    graph.add_edge("tool", "llm")
    return graph.compile()


def _clone_state(state: AgentState) -> AgentState:
    cloned: AgentState = {}
    if "messages" in state:
        cloned["messages"] = list(state["messages"])
    if "llm_input" in state:
        cloned["llm_input"] = list(state["llm_input"])
    if "pending_tool_calls" in state:
        cloned["pending_tool_calls"] = list(state["pending_tool_calls"])
    if "tool_calls" in state:
        cloned["tool_calls"] = [dict(item) for item in state["tool_calls"]]
    if "tool_invocations" in state:
        cloned["tool_invocations"] = [dict(item) for item in state["tool_invocations"]]
    if "raw_llm_response" in state:
        cloned["raw_llm_response"] = state["raw_llm_response"]
    return cloned


def _apply_delta(target: AgentState, delta: AgentState) -> None:
    for key, value in delta.items():
        if key == "messages":
            target.setdefault("messages", [])
            target["messages"].extend(value)
        elif key == "tool_invocations":
            existing = list(target.get("tool_invocations", []))
            existing.extend(value)
            target["tool_invocations"] = existing
        else:
            target[key] = value


class LangGraphAgent:
    """Thin wrapper around the compiled LangGraph application."""

    def __init__(self, llm: BaseChatModel, tools: Sequence[BaseTool]) -> None:
        self._llm = llm
        self._tools = tuple(tools)
        self._app = create_agent_app(llm, tools)

    async def run(
        self,
        messages: Sequence[BaseMessage],
        *,
        observer: Any | None = None,
    ) -> AgentState:
        initial_state: AgentState = {"messages": list(messages)}
        current_state: AgentState = _clone_state(initial_state)
        try:
            async for event in self._app.astream(initial_state, stream_mode="updates"):
                for node_name, delta in event.items():
                    before = _clone_state(current_state)
                    _apply_delta(current_state, delta)
                    if observer is not None:
                        await observer.record_node(node_name, before, _clone_state(current_state))
        finally:
            if observer is not None:
                try:
                    await observer.finalize(_clone_state(current_state))
                except Exception as exc:  # pragma: no cover - observability should not break the agent
                    logger.exception("Observer finalization failed: %s", exc)
        return current_state
