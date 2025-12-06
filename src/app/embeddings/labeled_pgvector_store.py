"""PGVector store extension that keeps Confluence labels in a dedicated column."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from sqlalchemy import insert, text

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.vector_stores.utils import node_to_metadata_dict
from llama_index.vector_stores.postgres import PGVectorStore


class LabeledPGVectorStore(PGVectorStore):
    """Adds a ``labels`` text[] column alongside the default pgvector schema."""

    _labels_column_ready: bool = PrivateAttr(default=False)
    _labels_column_name: str = PrivateAttr(default="labels")

    def __init__(self, *args: Any, labels_column: str = "labels", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._labels_column_name = labels_column

    def _initialize(self) -> None:
        super()._initialize()
        if not self._labels_column_ready:
            self._ensure_labels_column()

    def _ensure_labels_column(self) -> None:
        table_fqn = f"{self.schema_name}.{self._table_class.__tablename__}"
        with self._session() as session, session.begin():
            session.execute(
                text(
                    f"ALTER TABLE {table_fqn} "
                    f"ADD COLUMN IF NOT EXISTS {self._labels_column_name} TEXT[]"
                )
            )
            session.commit()
        self._labels_column_ready = True

    def _build_row_payload(self, node: BaseNode) -> Dict[str, Any]:
        metadata = node_to_metadata_dict(
            node,
            remove_text=True,
            flat_metadata=self.flat_metadata,
        )
        labels_raw = metadata.get("labels") if isinstance(metadata, dict) else None
        labels: List[str] | None
        if labels_raw is None:
            labels = None
        elif isinstance(labels_raw, list):
            labels = [str(label) for label in labels_raw if label]
        else:
            labels = [str(labels_raw)]
        return {
            "node_id": node.node_id,
            "embedding": node.get_embedding(),
            "text": node.get_content(metadata_mode=MetadataMode.NONE),
            "metadata_": metadata,
            self._labels_column_name: labels,
        }

    def add(self, nodes: Sequence[BaseNode], **add_kwargs: Any) -> List[str]:
        self._initialize()
        stmt = insert(self._table_class)
        ids: List[str] = []
        with self._session() as session, session.begin():
            for node in nodes:
                ids.append(node.node_id)
                row_payload = self._build_row_payload(node)
                session.execute(stmt, row_payload)
            session.commit()
        return ids

    async def async_add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> List[str]:
        self._initialize()
        stmt = insert(self._table_class)
        ids: List[str] = []
        async with self._async_session() as session, session.begin():
            for node in nodes:
                ids.append(node.node_id)
                row_payload = self._build_row_payload(node)
                await session.execute(stmt, row_payload)
            await session.commit()
        return ids
