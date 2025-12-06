"""Helpers for interacting with pgvector via LlamaIndex."""
from __future__ import annotations

from llama_index.vector_stores.postgres.base import PGType

from ..config import Settings
from .labeled_pgvector_store import LabeledPGVectorStore


def create_pgvector_store(settings: Settings) -> LabeledPGVectorStore:
    """Instantiate a pgvector-backed vector store with label support."""
    async_url = settings.async_db_url()
    sync_url = settings.sync_db_url()
    indexed_metadata: set[tuple[str, PGType]] = {("labels", "text[]")}
    return LabeledPGVectorStore(
        connection_string=sync_url,
        async_connection_string=async_url,
        table_name=settings.vector_collection,
        schema_name=settings.database_schema,
        embed_dim=settings.embedding_dim,
        indexed_metadata_keys=indexed_metadata,
    )
