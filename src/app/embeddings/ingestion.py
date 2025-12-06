"""Webhook ingestion pipeline for Confluence pages."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser

from ..config import Settings
from .markdown_utils import page_as_md
from .ollama import OllamaBgeM3Embedding
from .vector_store import create_pgvector_store

logger = logging.getLogger(__name__)


class PageIngestionService:
    """Coordinates Confluence fetch + LlamaIndex ingestion."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.embed_model = OllamaBgeM3Embedding(
            base_url=settings.ollama_base_url,
            model_name=settings.embedding_model_name,
            timeout=settings.request_timeout,
            max_retries=settings.embedding_max_retries,
            retry_backoff=settings.embedding_retry_backoff,
        )
        self.splitter = self._build_chunker()
        self.vector_store = create_pgvector_store(settings)

    def process_page(
        self,
        page_id: str,
        *,
        document_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        labels: Optional[List[str]] = None,
    ) -> None:
        """Fetch, chunk, embed, and store a Confluence page."""
        logger.info("Processing Confluence page %s", page_id)

        if document_text is None or metadata is None:
            logger.error(
                "process_page called without document_text/metadata for page %s; caller must supply original Confluence body and metadata",
                page_id,
            )
            raise ValueError("document_text and metadata are required")

        metadata = dict(metadata)

        # Convert storage HTML to markdown for ingestion
        document_text = page_as_md(document_text)

        metadata = metadata or {}
        metadata.setdefault("page_id", page_id)
        normalized_labels = self._normalize_labels(labels if labels is not None else metadata.get("labels"))
        metadata["labels"] = normalized_labels

        allowed = self.settings.allowed_spaces()
        space_key = metadata.get("space_key")
        if allowed and space_key not in allowed:
            logger.info(
                "Skipping page %s because space %s is not whitelisted",
                page_id,
                space_key,
            )
            return

        self._ingest_document(page_id, document_text or "", metadata)

    def _build_nodes(self, document: Document) -> List:
        """Chunk the document and ensure deterministic IDs."""
        nodes = self.splitter.get_nodes_from_documents([document])
        # Use page_id without version for consistent IDs across updates
        for idx, node in enumerate(nodes):
            node.id_ = f"{document.doc_id}:{idx}"
        return nodes

    def _build_chunker(self):
        if self.settings.use_semantic_chunker:
            logger.info(
                "Using SemanticSplitterNodeParser with buffer_size=%s breakpoint_percentile=%s",
                self.settings.semantic_chunker_buffer_size,
                self.settings.semantic_chunker_breakpoint_percentile,
            )
            return SemanticSplitterNodeParser.from_defaults(
                embed_model=self.embed_model,
                buffer_size=self.settings.semantic_chunker_buffer_size,
                breakpoint_percentile_threshold=self.settings.semantic_chunker_breakpoint_percentile,
            )

        logger.info(
            "Using SentenceSplitter with chunk_size=%s overlap=%s",
            self.settings.chunk_size,
            self.settings.chunk_overlap,
        )
        return SentenceSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

    def _delete_page_vectors(self, page_id: str) -> None:
        """Delete all existing vector embeddings for a given page."""
        try:
            # Delete vectors whose node IDs start with the page_id prefix
            self.vector_store.delete(f"{page_id}")
            logger.info("Deleted existing vectors for page %s", page_id)
        except Exception as e:
            logger.warning("Could not delete existing vectors for page %s: %s", page_id, e)

    def _ingest_document(self, page_id: str, document_text: str, metadata: Dict[str, Any]) -> None:
        if not document_text.strip():
            logger.warning("Page %s has no textual content to index", page_id)
            return

        self._delete_page_vectors(page_id)

        document = Document(text=document_text, metadata=metadata, id_=str(page_id))
        try:
            nodes = self._build_nodes(document)
        except Exception as e:
            logger.error("Failed to build nodes for page %s: %s", page_id, e, exc_info=True)
            raise
        if not nodes:
            logger.warning("Page %s produced zero nodes after chunking", page_id)
            return

        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        try:
            VectorStoreIndex(nodes, storage_context=storage_context, embed_model=self.embed_model)
        except Exception as e:
            logger.error("Failed to create vector index for page %s: %s", page_id, e, exc_info=True)
            raise
        logger.info("Finished indexing page %s (%s nodes)", page_id, len(nodes))

    @staticmethod
    def _normalize_labels(raw_labels: Optional[Any]) -> List[str]:
        if raw_labels is None:
            return []
        if isinstance(raw_labels, list):
            return [str(label) for label in raw_labels if label]
        return [str(raw_labels)]
