"""Retriever that queries pgvector directly and reranks results."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

from llama_index.core import QueryBundle
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryResult

from ..config import Settings
from ..config.db import fetch_scalar
from ..embeddings.ollama import OllamaBgeM3Embedding
from ..embeddings.vector_store import create_pgvector_store

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Container holding reranked nodes and raw search hits."""

    reranked_nodes: Sequence[NodeWithScore]
    raw_hits: Sequence[NodeWithScore]


class RetrieverService:
    """Issue ANN queries against pgvector and apply cross-encoder reranking."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.embed_model = OllamaBgeM3Embedding(
            base_url=settings.ollama_base_url,
            model_name=settings.embedding_model_name,
            timeout=settings.request_timeout,
            max_retries=settings.embedding_max_retries,
            retry_backoff=settings.embedding_retry_backoff,
        )
        self.vector_store = create_pgvector_store(settings)
        self._reranker = self._init_reranker()
        self._cached_count: Optional[int] = None

    # ------------------------------------------------------------------
    def is_ready(self) -> bool:
        """Return True once we've verified that the vector store has data."""

        return bool(self._cached_count) and self._cached_count > 0

    async def refresh(self) -> int:
        """Check the pgvector table for available nodes and cache the count."""

        count = await self._count_nodes()
        if count == 0:
            raise RuntimeError("No nodes found in the vector store; run ingestion first.")
        self._cached_count = count
        logger.info("Retriever ready with %s nodes in pgvector", count)
        return count

    async def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        """Fetch top-K results from pgvector and rerank them."""

        if not self.is_ready():
            try:
                await self.refresh()
            except RuntimeError:
                return RetrievalResult(reranked_nodes=[], raw_hits=[])

        desired_top_k = top_k or self.settings.retriever_top_k
        search_k = max(self.settings.retriever_search_k, desired_top_k)

        query_embedding = self.embed_model._get_query_embedding(query)
        result = self.vector_store.query(
            VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=search_k)
        )
        raw_hits = self._nodes_from_result(result)
        if not raw_hits:
            return RetrievalResult(reranked_nodes=[], raw_hits=[])

        reranked = self._apply_reranker(raw_hits, query, desired_top_k)
        return RetrievalResult(reranked_nodes=reranked, raw_hits=raw_hits[:search_k])

    # ------------------------------------------------------------------
    def _nodes_from_result(self, result: VectorStoreQueryResult) -> List[NodeWithScore]:
        nodes = list(result.nodes or [])
        similarities = result.similarities or []

        scored_nodes: List[NodeWithScore] = []
        for idx, node in enumerate(nodes):
            score = float(similarities[idx]) if idx < len(similarities) else 0.0
            if isinstance(node, NodeWithScore):
                scored_nodes.append(NodeWithScore(node=node.node, score=score))
            else:
                scored_nodes.append(NodeWithScore(node=node, score=score))
        return scored_nodes

    def _apply_reranker(
        self, nodes: Sequence[NodeWithScore], query: str, desired_top_k: int
    ) -> Sequence[NodeWithScore]:
        if not nodes:
            return nodes
        if not self._reranker:
            return nodes[:desired_top_k]
        query_bundle = QueryBundle(query_str=query)
        reranked = self._reranker.postprocess_nodes(list(nodes), query_bundle)
        top_n = min(self.settings.reranker_top_n, desired_top_k, len(reranked))
        return reranked[:top_n]

    def _init_reranker(self):
        return SentenceTransformerRerank(
            model=self.settings.reranker_model_name,
            top_n=self.settings.reranker_top_n,
        )

    async def _count_nodes(self) -> int:
        settings = self.settings
        schema = settings.database_schema
        table = settings.vector_collection_with_prefix
        query = f"SELECT COUNT(*) FROM {schema}.{table}"
        result = await fetch_scalar(query, settings)
        return int(result) if result is not None else 0
