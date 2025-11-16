"""Custom middleware utilities for the FastAPI app."""
from __future__ import annotations

import logging
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class RawRequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs the raw request body for every incoming request."""

    def __init__(self, app, max_bytes: int = 65536):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: Callable[[Request], Response]) -> Response:
        body = await request.body()
        truncated = body[: self.max_bytes]
        body_text = truncated.decode("utf-8", errors="replace")
        if len(body) > self.max_bytes:
            body_text += "... [truncated]"
        logger.info("Raw request %s %s body: %s", request.method, request.url.path, body_text)

        async def receive() -> dict:
            return {"type": "http.request", "body": body, "more_body": False}

        request._receive = receive  # type: ignore[attr-defined]
        response = await call_next(request)
        return response
