"""Conversation orchestration that combines RAG and LangChain prompts."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from llama_index.core.schema import NodeWithScore

from ..config import Settings
from ..conversation_history import ConversationHistoryStore, ConversationMessage
from ..llm import create_chat_model
from ..retriever.service import RetrieverService
from ..observability import create_langfuse_observer
from .graph import build_chat_workflow
from .state import ChatState


@dataclass
class ChatResult:
    session_id: str
    response: str
    sources: Sequence[NodeWithScore]
    raw_hits: Sequence[NodeWithScore]
    context: str


class ChatbotService:
    """Generate chat responses with retrieval-augmented generation."""

    def __init__(
        self,
        settings: Settings,
        retriever: RetrieverService,
        history_store: Optional[ConversationHistoryStore] = None,
    ) -> None:
        self._settings = settings
        self._retriever = retriever
        self._history_store = history_store or ConversationHistoryStore(settings)
        self._model = create_chat_model(settings)
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_prompt}"),
                ("system", "Enterprise knowledge context:\n{context}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )
        self._chain = self._prompt | self._model | StrOutputParser()
        self._workflow = build_chat_workflow(self)

    @property
    def settings(self) -> Settings:
        return self._settings

    async def generate_response(
        self,
        *,
        session_id: str,
        user_message: str,
        top_k: Optional[int] = None,
    ) -> ChatResult:
        previous_history = await self._history_store.fetch_recent_messages(
            session_id, self._settings.conversation_history_max_messages
        )
        history_messages = self._to_langchain_messages(previous_history)

        await self._history_store.add_message(session_id, "user", user_message)

        observer = create_langfuse_observer(
            self._settings,
            session_id=session_id,
            user_message=user_message,
        )

        initial_state: ChatState = {
            "session_id": session_id,
            "user_message": user_message,
            "top_k": top_k,
            "history_messages": history_messages,
        }
        if observer is not None:
            initial_state["observer"] = observer

        result_state = await self._workflow.ainvoke(initial_state)

        response_text = (result_state.get("response") or "").strip()
        sources = result_state.get("sources") or []
        raw_hits = result_state.get("raw_hits") or []
        context = result_state.get("context") or ""

        if observer is not None:
            combined_state: Dict[str, Any] = {
                **{k: v for k, v in initial_state.items() if k != "observer"},
                **{k: v for k, v in result_state.items() if k != "observer"},
            }
            await observer.finalize(combined_state)

        return ChatResult(
            session_id=session_id,
            response=response_text,
            sources=sources,
            raw_hits=raw_hits,
            context=context,
        )

    async def _invoke_chain(self, inputs: Dict[str, object]) -> str:
        try:
            return await self._chain.ainvoke(inputs)
        except (AttributeError, NotImplementedError):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: self._chain.invoke(inputs))

    def _to_langchain_messages(self, history: Sequence[ConversationMessage]) -> List[BaseMessage]:
        messages: List[BaseMessage] = []
        for item in history:
            if item.role == "user":
                messages.append(HumanMessage(content=item.content))
            elif item.role == "assistant":
                messages.append(AIMessage(content=item.content))
            elif item.role == "system":
                messages.append(SystemMessage(content=item.content))
        return messages

    def _format_context(self, nodes: Sequence[NodeWithScore]) -> str:
        if not nodes:
            return ""
        max_chars = self._settings.rag_context_max_chars_per_source
        rendered: List[str] = []
        for idx, node_with_score in enumerate(nodes, start=1):
            node = node_with_score.node
            try:
                text = node.get_content()  # type: ignore[attr-defined]
            except AttributeError:
                text = getattr(node, "text", "")
            text = (text or "").strip()
            if max_chars and len(text) > max_chars:
                text = text[: max_chars - 3].rstrip() + "..."
            metadata = getattr(node, "metadata", None)
            metadata_str = ""
            if metadata:
                if isinstance(metadata, dict):
                    pairs = [f"{key}={value}" for key, value in metadata.items() if value is not None]
                else:
                    pairs = [f"{key}={value}" for key, value in dict(metadata).items() if value is not None]
                if pairs:
                    metadata_str = " | ".join(pairs)
            snippet = f"[Source {idx}] score={node_with_score.score:.3f}\n{text}"
            if metadata_str:
                snippet = f"{snippet}\nMetadata: {metadata_str}"
            rendered.append(snippet)
        return "\n\n".join(rendered)

