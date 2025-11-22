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
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: Optional[str] = None
    langfuse_environment: str = "production"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @model_validator(mode="after")
    def _ensure_database_url(self) -> "Settings":
        if not self.database_url:
            raise ValueError("Set DATABASE_URL in the environment.")
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
        if bool(self.langfuse_public_key) ^ bool(self.langfuse_secret_key):
            raise ValueError("Provide both LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY or neither")
        return self

    def base_db_url(self) -> str:
        """Return the base Postgres URL from the environment."""
        assert self.database_url  # ensured via validator
        return self.database_url


@lru_cache()
def get_settings() -> Settings:
    """Cached settings accessor."""
    return Settings()  # type: ignore[call-arg]
