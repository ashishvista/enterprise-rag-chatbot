"""LlamaIndex embedding wrapper for Ollama's bge-m3 model."""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from pathlib import Path
from typing import Awaitable, Callable, List

import httpx
from llama_index.core.embeddings import BaseEmbedding

from ..config import create_async_httpx_client, create_httpx_client

logger = logging.getLogger(__name__)


class OllamaBgeM3Embedding(BaseEmbedding):
    """Custom embedding class that calls a running Ollama server."""

    base_url: str
    model_name: str = "bge-m3"
    timeout: int = 30
    max_retries: int = 3
    retry_backoff: float = 0.5
    failure_log_path: Path = Path("ollama_failed_payloads.log")

    def __init__(
        self,
        base_url: str,
        model_name: str = "bge-m3",
        timeout: int = 30,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
        failure_log_path: str | Path | None = None,
    ):
        super().__init__(
            base_url=base_url.rstrip("/"),
            model_name=model_name,
            timeout=timeout,
        )
        object.__setattr__(self, "max_retries", max(0, max_retries))
        object.__setattr__(self, "retry_backoff", max(0.0, retry_backoff))
        if failure_log_path:
            object.__setattr__(self, "failure_log_path", Path(failure_log_path))

    # ---- sync helpers -------------------------------------------------
    def _embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        with create_httpx_client(base_url=self.base_url, timeout=self.timeout) as client:
            return [self._embed_single_sync(client, text) for text in texts]

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
        async with create_async_httpx_client(base_url=self.base_url, timeout=self.timeout) as client:
            return [await self._embed_single_async(client, text) for text in texts]

    async def _aembed_query(self, text: str) -> List[float]:
        vectors = await self._aembed([text])
        return vectors[0]

    async def _aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await self._aembed(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await self._aembed_query(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return await self._aembed_query(text)

    # ---- helpers -----------------------------------------------------
    def _embed_single_sync(self, client: httpx.Client, text: str) -> List[float]:
        payload = self._build_payload(text)

        def _do_request() -> httpx.Response:
            response = client.post("/api/embeddings", json=payload)
            response.raise_for_status()
            return response

        response = self._retry_sync(_do_request, payload)
        return self._extract_vector(response.json())

    async def _embed_single_async(self, client: httpx.AsyncClient, text: str) -> List[float]:
        payload = self._build_payload(text)

        async def _do_request() -> httpx.Response:
            response = await client.post("/api/embeddings", json=payload)
            response.raise_for_status()
            return response

        response = await self._retry_async(_do_request, payload)
        data = response.json()
        return self._extract_vector(data)

    def _build_payload(self, text: str) -> dict:
        return {"model": self.model_name, "prompt": text}

    def _extract_vector(self, data: dict) -> List[float]:
        if "embedding" in data:
            return data["embedding"]
        if "data" in data:
            items = data["data"] or []
            if items and "embedding" in items[0]:
                return items[0]["embedding"]
        raise ValueError(f"Ollama returned no embedding vectors: {data}")

    def _retry_sync(self, fn: Callable[[], httpx.Response], payload: dict) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return fn()
            except httpx.HTTPError as exc:  # transient Ollama server error
                last_error = exc
                if attempt == self.max_retries:
                    self._record_failed_payload(payload, exc)
                    raise
                time.sleep(self._backoff_delay(attempt))
        assert last_error is not None  # defensive: loop must either return or raise
        raise last_error

    async def _retry_async(self, fn: Callable[[], Awaitable[httpx.Response]], payload: dict) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await fn()
            except httpx.HTTPError as exc:  # transient Ollama server error
                last_error = exc
                if attempt == self.max_retries:
                    self._record_failed_payload(payload, exc)
                    raise
                await asyncio.sleep(self._backoff_delay(attempt))
        assert last_error is not None
        raise last_error

    def _backoff_delay(self, attempt: int) -> float:
        if self.retry_backoff == 0:
            return 0.0
        base_delay = self.retry_backoff * (2 ** attempt)
        return base_delay + random.uniform(0, self.retry_backoff)

    def _record_failed_payload(self, payload: dict, error: Exception) -> None:
        entry = {
            "timestamp": time.time(),
            "model": self.model_name,
            "payload": payload,
            "error": repr(error),
        }
        try:
            path = self.failure_log_path
            if path.parent and not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
        except Exception:  # pragma: no cover - best-effort logging
            logger.exception("Failed to record Ollama payload after retries exhausted")
