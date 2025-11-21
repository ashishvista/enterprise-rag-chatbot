"""LangGraph workflow definition for the chatbot service."""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict

from langgraph.graph import END, START, StateGraph

from .state import ChatState

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from .service import ChatbotService


def build_chat_workflow(service: "ChatbotService"):
    """Return a compiled LangGraph workflow bound to the provided service."""

    graph = StateGraph(ChatState)

    async def retrieve_context(state: ChatState) -> ChatState:
        top_k = state.get("top_k")
        retrieval = await service._retriever.retrieve(state["user_message"], top_k)
        sources = list(retrieval.reranked_nodes or [])
        raw_hits = list(retrieval.raw_hits)
        if not sources and raw_hits:
            sources = raw_hits
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
            "context": state.get("context") or "No enterprise context retrieved.",
            "history": state.get("history_messages", []),
            "question": state["user_message"],
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

    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("run_llm", run_llm)
    graph.add_node("store_response", store_response)
    graph.add_edge(START, "retrieve_context")
    graph.add_edge("retrieve_context", "run_llm")
    graph.add_edge("run_llm", "store_response")
    graph.add_edge("store_response", END)

    return graph.compile()
