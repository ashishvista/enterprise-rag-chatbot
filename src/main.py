"""Entry point for the FastAPI Confluence webhook service."""
from __future__ import annotations

from fastapi import FastAPI

from app.config import get_settings
from app.webhook import router as webhook_router


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Enterprise RAG Webhooks", version="0.1.0")
    app.include_router(webhook_router)

    @app.get("/health", tags=["health"])
    def healthcheck() -> dict:
        return {
            "status": "ok",
            "vector_collection": settings.vector_collection,
            "embedding_model": settings.embedding_model_name,
        }

    return app


app = create_app()
