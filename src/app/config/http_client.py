"""Centralized helpers for constructing HTTPX clients."""
from __future__ import annotations

from typing import Any, Optional

import httpx


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def create_httpx_client(
    *,
    base_url: str,
    timeout: Optional[float | int | httpx.Timeout] = None,
    **kwargs: Any,
) -> httpx.Client:
    """Return a configured synchronous httpx.Client."""
    return httpx.Client(base_url=_normalize_base_url(base_url), timeout=timeout, **kwargs)


def create_async_httpx_client(
    *,
    base_url: str,
    timeout: Optional[float | int | httpx.Timeout] = None,
    **kwargs: Any,
) -> httpx.AsyncClient:
    """Return a configured asynchronous httpx.AsyncClient."""
    return httpx.AsyncClient(base_url=_normalize_base_url(base_url), timeout=timeout, **kwargs)
