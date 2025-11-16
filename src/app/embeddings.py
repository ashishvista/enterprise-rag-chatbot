"""LlamaIndex embedding wrapper for Ollama's bge-m3 model."""
from __future__ import annotations

from typing import List

import httpx
from llama_index.core.embeddings import BaseEmbedding


class OllamaBgeM3Embedding(BaseEmbedding):
    """Custom embedding class that calls a running Ollama server."""

    base_url: str
    model_name: str = "bge-m3"
    timeout: int = 30

    def __init__(self, base_url: str, model_name: str = "bge-m3", timeout: int = 30):
        super().__init__(
            base_url=base_url.rstrip("/"),
            model_name=model_name,
            timeout=timeout,
        )

    # ---- sync helpers -------------------------------------------------
    def _embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        payload = {"model": self.model_name, "input": texts}
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            response = client.post("/api/embeddings", json=payload)
            response.raise_for_status()
            data = response.json()
        vectors = data.get("data")
        if not vectors:
            raise ValueError("Ollama returned no embedding vectors")
        return [chunk["embedding"] for chunk in vectors]

    def _embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]

    def _embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    # Compatibility with newer BaseEmbedding hooks
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed_query(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed_query(text)

    # ---- async helpers ------------------------------------------------
    async def _aembed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        payload = {"model": self.model_name, "input": texts}
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
            response = await client.post("/api/embeddings", json=payload)
            response.raise_for_status()
            data = response.json()
        vectors = data.get("data")
        if not vectors:
            raise ValueError("Ollama returned no embedding vectors")
        return [chunk["embedding"] for chunk in vectors]

    async def _aembed_query(self, text: str) -> List[float]:
        vectors = await self._aembed([text])
        return vectors[0]

    async def _aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await self._aembed(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await self._aembed_query(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await self._aembed_query(text)
