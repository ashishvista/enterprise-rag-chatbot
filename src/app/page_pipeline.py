"""Webhook ingestion pipeline for Confluence pages."""
from __future__ import annotations

import logging
from typing import Dict, List

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

from .config import Settings
from .confluence import ConfluenceClient
from .embeddings import OllamaBgeM3Embedding
from .vector_store import create_pgvector_store

logger = logging.getLogger(__name__)


class PageIngestionService:
    """Coordinates Confluence fetch + LlamaIndex ingestion."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.splitter = SentenceSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self.embed_model = OllamaBgeM3Embedding(
            base_url=settings.ollama_base_url,
            model_name=settings.embedding_model_name,
            timeout=settings.request_timeout,
        )
        self.vector_store = create_pgvector_store(settings)

    def process_page(self, page_id: str) -> None:
        """Fetch, chunk, embed, and store a Confluence page."""
        logger.info("Processing Confluence page %s", page_id)
        with ConfluenceClient(self.settings) as client:
            page_payload = client.fetch_page(page_id)
        metadata = ConfluenceClient.page_metadata(page_payload)
        allowed = self.settings.allowed_spaces()
        space_key = metadata.get("space_key")
        if allowed and space_key not in allowed:
            logger.info("Skipping page %s because space %s is not whitelisted", page_id, space_key)
            return
        document_text = ConfluenceClient.page_as_text(page_payload)
        if not document_text.strip():
            logger.warning("Page %s has no textual content to index", page_id)
            return
        document = Document(text=document_text, metadata=metadata, id_=str(page_id))
        nodes = self._build_nodes(document)
        if not nodes:
            logger.warning("Page %s produced zero nodes after chunking", page_id)
            return
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        VectorStoreIndex(nodes, storage_context=storage_context, embed_model=self.embed_model)
        logger.info("Finished indexing page %s (%s nodes)", page_id, len(nodes))

    def _build_nodes(self, document: Document) -> List:
        """Chunk the document and ensure deterministic IDs."""
        nodes = self.splitter.get_nodes_from_documents([document])
        version = document.metadata.get("version", "0")
        for idx, node in enumerate(nodes):
            node.id_ = f"{document.doc_id}:{version}:{idx}"
        return nodes
