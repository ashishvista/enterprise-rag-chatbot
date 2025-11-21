"""Dependency helpers for access to the RetrieverService."""
from __future__ import annotations

import json
from typing import Optional, Tuple

from fastapi import Depends

from ..config import Settings, get_settings
from .service import RetrieverService

_retriever_cache: Optional[Tuple[str, RetrieverService]] = None


def get_retriever_service(settings: Settings = Depends(get_settings)) -> RetrieverService:
    """Return a cached RetrieverService keyed by the settings signature."""
    global _retriever_cache
    signature = json.dumps(settings.model_dump(mode="json"), sort_keys=True)
    if _retriever_cache is None or _retriever_cache[0] != signature:
        _retriever_cache = (signature, RetrieverService(settings))
    return _retriever_cache[1]
