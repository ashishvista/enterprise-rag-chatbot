"""Helpers for interacting with pgvector via LlamaIndex."""
from __future__ import annotations

from llama_index.vector_stores.postgres import PGVectorStore

from ..config import Settings
from ..config.db import to_async_sqlalchemy_url, to_sync_sqlalchemy_url


def create_pgvector_store(settings: Settings) -> PGVectorStore:
    """Instantiate a pgvector-backed vector store."""
    base_url = settings.base_db_url()
    async_url = to_async_sqlalchemy_url(base_url)
    sync_url = to_sync_sqlalchemy_url(base_url)
    return PGVectorStore(
        connection_string=sync_url,
        async_connection_string=async_url,
        table_name=settings.vector_collection,
        schema_name=settings.database_schema,
        embed_dim=settings.embedding_dim,
    )
