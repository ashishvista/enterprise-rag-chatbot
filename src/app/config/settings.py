"""Configuration helpers for the webhook service."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Env-backed settings."""

    confluence_base_url: str
    confluence_username: str
    confluence_api_token: str
    confluence_space_whitelist: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    embedding_model_name: str = "bge-m3"
    embedding_dim: int = 1024  # bge-m3: 1024, qwen3-embedding:8b: 4096
    embedding_max_retries: int = 5
    embedding_retry_backoff: float = 0.5
    llm_model_name: str = "gpt_oss"
    llm_temperature: float = 0.1
    llm_max_output_tokens: Optional[int] = None
    llm_context_window: Optional[int] = None
    chat_system_prompt: str = (
        "You are an enterprise-ready assistant. Always ground answers in the provided "
        "context. If the context does not contain the answer, say you do not know."
    )
    conversation_history_table: str = "chatbot_conversation_history"
    conversation_history_max_messages: int = 20
    rag_context_max_chars_per_source: int = 1200
    use_semantic_chunker: bool = False
    semantic_chunker_buffer_size: int = 1
    semantic_chunker_breakpoint_percentile: int = 95
    database_url_async: Optional[str] = None
    database_url: Optional[str] = None
    database_schema: str = "public"
    vector_collection: str = "confluence_pages"
    vector_collection_with_prefix: str = "data_confluence_pages"
    chunk_size: int = 1024
    chunk_overlap: int = 100
    request_timeout: int = 30
    retriever_top_k: int = 5
    retriever_search_k: int = 15
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"
    reranker_top_n: int = 3

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @model_validator(mode="after")
    def _ensure_database_urls(self) -> "Settings":
        """Guarantee at least one DB URL is present and derive/normalize the other."""
        if not self.database_url_async and not self.database_url:
            raise ValueError("Set DATABASE_URL_ASYNC or DATABASE_URL in the environment.")

        if self.database_url:
            self.database_url = normalize_sync_connection_string(self.database_url)
        if self.database_url_async:
            self.database_url_async = normalize_async_connection_string(self.database_url_async)

        if not self.database_url_async and self.database_url:
            self.database_url_async = infer_async_connection_string(self.database_url)
        if not self.database_url and self.database_url_async:
            self.database_url = infer_sync_connection_string(self.database_url_async)
        return self

    def allowed_spaces(self) -> Optional[list[str]]:
        """Return a list of allowed Confluence space keys, if configured."""
        if not self.confluence_space_whitelist:
            return None
        return [space.strip() for space in self.confluence_space_whitelist.split(",") if space.strip()]

    @model_validator(mode="after")
    def _validate_retriever_settings(self) -> "Settings":
        if self.retriever_top_k <= 0:
            raise ValueError("retriever_top_k must be positive")
        if self.retriever_search_k < self.retriever_top_k:
            raise ValueError("retriever_search_k must be >= retriever_top_k")
        if self.reranker_top_n <= 0:
            raise ValueError("reranker_top_n must be positive")
        if self.reranker_top_n > self.retriever_search_k:
            raise ValueError("reranker_top_n cannot exceed retriever_search_k")
        if not self.vector_collection_with_prefix:
            raise ValueError("vector_collection_with_prefix must be set")
        if self.conversation_history_max_messages <= 0:
            raise ValueError("conversation_history_max_messages must be positive")
        if self.rag_context_max_chars_per_source <= 0:
            raise ValueError("rag_context_max_chars_per_source must be positive")
        return self

    def async_db_url(self) -> str:
        """Return the asyncpg connection string."""
        assert self.database_url_async  # ensured via validator
        return self.database_url_async

    def sync_db_url(self) -> str:
        """Return a psycopg connection string, deriving from async if needed."""
        assert self.database_url  # ensured via validator
        return self.database_url


@lru_cache()
def get_settings() -> Settings:
    """Cached settings accessor."""
    return Settings()  # type: ignore[call-arg]


def infer_sync_connection_string(async_url: str) -> str:
    """Best-effort conversion from asyncpg URL to psycopg equivalent."""
    return normalize_sync_connection_string(async_url)


def infer_async_connection_string(sync_url: str) -> str:
    """Best-effort conversion from psycopg URL to asyncpg equivalent."""
    return normalize_async_connection_string(sync_url)


def normalize_sync_connection_string(url: str) -> str:
    """Ensure the SQLAlchemy URL uses the psycopg (sync) driver."""
    if "+psycopg" in url:
        return url
    if "+asyncpg" in url:
        return url.replace("+asyncpg", "+psycopg", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def normalize_async_connection_string(url: str) -> str:
    """Ensure the SQLAlchemy URL uses the asyncpg driver."""
    if "+asyncpg" in url:
        return url
    if "+psycopg" in url:
        return url.replace("+psycopg", "+asyncpg", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url
