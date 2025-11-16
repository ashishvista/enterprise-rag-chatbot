"""Configuration helpers for the webhook service."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Env-backed settings."""

    confluence_base_url: str
    confluence_username: str
    confluence_api_token: str
    confluence_space_whitelist: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    embedding_model_name: str = "bge-m3"
    database_url: str
    vector_collection: str = "confluence_pages"
    chunk_size: int = 1024
    chunk_overlap: int = 100
    request_timeout: int = 30

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    def allowed_spaces(self) -> Optional[list[str]]:
        """Return a list of allowed Confluence space keys, if configured."""
        if not self.confluence_space_whitelist:
            return None
        return [space.strip() for space in self.confluence_space_whitelist.split(",") if space.strip()]


@lru_cache()
def get_settings() -> Settings:
    """Cached settings accessor."""
    return Settings()  # type: ignore[call-arg]
