"""Helpers for interacting with pgvector via LlamaIndex."""
from __future__ import annotations

from llama_index.vector_stores.postgres import PGVectorStore

from ..config import Settings


def create_pgvector_store(settings: Settings) -> PGVectorStore:
    """Instantiate a pgvector-backed vector store."""
    async_url = settings.async_db_url()
    sync_url = settings.sync_db_url()
    return PGVectorStore(
        connection_string=sync_url,
        async_connection_string=async_url,
        table_name=settings.vector_collection,
        schema_name=settings.database_schema,
        embed_dim=1024,  # bge-m3 outputs 1024-d vectors
    )
