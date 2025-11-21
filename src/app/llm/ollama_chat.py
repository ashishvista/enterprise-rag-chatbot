"""LangChain chat model factory for Ollama-backed models."""
from __future__ import annotations

from typing import Any, Dict

from langchain_ollama import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel

from ..config import Settings


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
    return ChatOllama(**params)
