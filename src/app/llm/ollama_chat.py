"""LangChain chat model factory for Ollama-backed models."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult
from langchain_core.runnables.config import RunnableConfig
from langchain_ollama import ChatOllama
from langchain_community.llms.ollama import OllamaEndpointNotFoundError

from ..config import Settings


class SafeChatOllama(ChatOllama):
    """ChatOllama variant that raises a clear runtime error when model is missing."""

    def __init__(self, *, missing_model_message: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._missing_model_message = missing_model_message

    def _handle_missing_model(self, error: Exception) -> None:
        raise RuntimeError(self._missing_model_message) from error

    def _generate(
        self,
        messages: List[Any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except OllamaEndpointNotFoundError as exc:
            self._handle_missing_model(exc)

    async def _agenerate(
        self,
        messages: List[Any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            if "config" in kwargs:
                kwargs = dict(kwargs)
                kwargs.pop("config", None)
            return await super()._agenerate(
                messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
        except OllamaEndpointNotFoundError as exc:
            self._handle_missing_model(exc)


def create_chat_model(settings: Settings) -> BaseChatModel:
    """Instantiate a LangChain ChatOllama model from settings."""
    params: Dict[str, Any] = {
        "base_url": settings.ollama_base_url.rstrip("/"),
        "model": settings.llm_model_name,
        "temperature": settings.llm_temperature,
    }
    if settings.llm_max_output_tokens is not None:
        params["num_predict"] = settings.llm_max_output_tokens
    if settings.llm_context_window is not None:
        params["num_ctx"] = settings.llm_context_window
    missing_message = (
        f"Ollama model '{settings.llm_model_name}' is unavailable. "
        f"Pull it with 'ollama pull {settings.llm_model_name}' or update LLM_MODEL_NAME."
    )
    return SafeChatOllama(missing_model_message=missing_message, **params)
