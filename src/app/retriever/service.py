"""Retriever that queries pgvector directly and reranks results."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from llama_index.core import QueryBundle
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    FilterCondition,
    FilterOperator,
    VectorStoreQuery,
    VectorStoreQueryResult,
)

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


@dataclass
class SerializedNode:
    """Normalized representation of a retrieved node for external consumers."""

    node_id: str
    score: float
    text: str
    metadata: Optional[Dict[str, Any]] = None


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

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        labels: Optional[List[str]] = None,
    ) -> RetrievalResult:
        """Fetch top-K results from pgvector and rerank them, with optional label filter."""

        if not self.is_ready():
            try:
                await self.refresh()
            except RuntimeError:
                return RetrievalResult(reranked_nodes=[], raw_hits=[])

        desired_top_k = top_k or self.settings.retriever_top_k
        search_k = max(self.settings.retriever_search_k, desired_top_k)

        query_embedding = self.embed_model._get_query_embedding(query)
        metadata_filters = self._build_label_filters(labels)
        result = self.vector_store.query(
            VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=search_k,
                filters=metadata_filters,
            )
        )
        raw_hits = self._nodes_from_result(result)
        raw_hits = self._filter_by_score(raw_hits, self.settings.retriever_min_score)
        if not raw_hits:
            return RetrievalResult(reranked_nodes=[], raw_hits=[])

        reranked = self._apply_reranker(raw_hits, query, desired_top_k)
        reranked = list(self._filter_by_score(reranked, self.settings.reranker_min_score))
        raw_hits_sliced = raw_hits[:search_k]
        return RetrievalResult(reranked_nodes=reranked, raw_hits=raw_hits_sliced)

    # ------------------------------------------------------------------
    def serialize_node(self, node_with_score: NodeWithScore) -> SerializedNode:
        node = node_with_score.node
        node_id = getattr(node, "node_id", None) or getattr(node, "id_", None) or getattr(node, "doc_id", None)
        try:
            text = node.get_content()  # type: ignore[attr-defined]
        except AttributeError:
            text = getattr(node, "text", "") or ""
        metadata = getattr(node, "metadata", None)
        if metadata is None:
            metadata_dict: Optional[Dict[str, Any]] = None
        elif isinstance(metadata, dict):
            metadata_dict = metadata
        else:
            metadata_dict = dict(metadata)
        score = float(node_with_score.score) if node_with_score.score is not None else 0.0
        return SerializedNode(
            node_id=str(node_id or ""),
            score=score,
            text=text,
            metadata=metadata_dict,
        )

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

    def _filter_by_score(
        self, nodes: Sequence[NodeWithScore], threshold: Optional[float]
    ) -> List[NodeWithScore]:
        if threshold is None:
            return list(nodes)
        filtered: List[NodeWithScore] = []
        for node_with_score in nodes:
            score = getattr(node_with_score, "score", None)
            if score is None or score >= threshold:
                filtered.append(node_with_score)
        return filtered

    def _init_reranker(self):
        return SentenceTransformerRerank(
            model=self.settings.reranker_model_name,
            top_n=self.settings.reranker_top_n,
        )

    def _build_label_filters(
        self, labels: Optional[List[str]]
    ) -> Optional[MetadataFilters]:
        if not labels:
            return None
        cleaned = [label for label in labels if label]
        if not cleaned:
            return None
        # Use ANY operator on labels array column
        return MetadataFilters(
            filters=[
                MetadataFilter(
                    key="labels",
                    value=cleaned,
                    operator=FilterOperator.ANY,
                )
            ],
            condition=FilterCondition.AND,
        )

    async def _count_nodes(self) -> int:
        settings = self.settings
        schema = settings.database_schema
        table = settings.vector_collection_with_prefix
        query = f"SELECT COUNT(*) FROM {schema}.{table}"
        result = await fetch_scalar(query, settings)
        return int(result) if result is not None else 0
