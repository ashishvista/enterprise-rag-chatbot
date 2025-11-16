"""Helpers for interacting with pgvector via LlamaIndex."""
from __future__ import annotations

from llama_index.vector_stores.postgres import PGVectorStore

from .config import Settings


def create_pgvector_store(settings: Settings) -> PGVectorStore:
    """Instantiate a pgvector-backed vector store."""
    return PGVectorStore(
        connection_string=settings.database_url,
        table_name=settings.vector_collection,
        embed_dim=None,  # let pgvector infer from first insert
    )
