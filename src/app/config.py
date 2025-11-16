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
    database_url_async: Optional[str] = None
    database_url: Optional[str] = None
    database_schema: str = "public"
    vector_collection: str = "confluence_pages"
    chunk_size: int = 1024
    chunk_overlap: int = 100
    request_timeout: int = 30

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @model_validator(mode="after")
    def _ensure_database_urls(self) -> "Settings":
        """Guarantee at least one DB URL is present and derive the other."""
        if not self.database_url_async and not self.database_url:
            raise ValueError("Set DATABASE_URL_ASYNC or DATABASE_URL in the environment.")
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
    if "+asyncpg" in async_url:
        return async_url.replace("+asyncpg", "+psycopg", 1)
    if async_url.startswith("postgresql://"):
        return async_url.replace("postgresql://", "postgresql+psycopg://", 1)
    raise ValueError(
        "Unable to infer sync connection string from DATABASE_URL_ASYNC. "
        "Set DATABASE_URL explicitly."
    )


def infer_async_connection_string(sync_url: str) -> str:
    """Best-effort conversion from psycopg URL to asyncpg equivalent."""
    if "+psycopg" in sync_url:
        return sync_url.replace("+psycopg", "+asyncpg", 1)
    if sync_url.startswith("postgresql://"):
        return sync_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    raise ValueError(
        "Unable to infer async connection string from DATABASE_URL. "
        "Set DATABASE_URL_ASYNC explicitly."
    )
