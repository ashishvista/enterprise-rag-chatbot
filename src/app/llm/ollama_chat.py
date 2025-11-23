"""LangChain chat model factory for Ollama-backed models."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult
from langchain_core.runnables.config import RunnableConfig
from langchain_ollama import ChatOllama
from langchain_community.llms.ollama import OllamaEndpointNotFoundError

from ..config import Settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class SafeChatOllama(ChatOllama):
    """ChatOllama variant that raises a clear runtime error when model is missing."""

    def __init__(self, *, missing_model_message: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._missing_model_message = missing_model_message

    def _handle_missing_model(self, error: Exception) -> None:
        raise RuntimeError(self._missing_model_message) from error

    def _log_result(self, result: ChatResult) -> None:
        try:
            messages = []
            for generation in result.generations:
                text = getattr(generation, "text", None)
                if text:
                    messages.append(text)
                    continue
                message = getattr(generation, "message", None)
                content = getattr(message, "content", None) if message is not None else None
                if content:
                    messages.append(str(content))
            if messages:
                logger.info("Ollama raw response: %s", messages[0])
        except Exception:  # pragma: no cover - logging should not break inference
            logger.exception("Failed to log Ollama response")

    def _generate(
        self,
        messages: List[Any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            self._log_result(result)
            return result
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
            result = await super()._agenerate(
                messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
            self._log_result(result)
            return result
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
