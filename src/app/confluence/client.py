"""Confluence REST helpers."""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..config import Settings, create_httpx_client

class ConfluenceClient:
    """Minimal Confluence Cloud REST client."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = create_httpx_client(
            base_url=settings.confluence_base_url,
            auth=(settings.confluence_username, settings.confluence_api_token),
            timeout=settings.request_timeout,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ConfluenceClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        self.close()

    def fetch_page(self, page_id: str) -> Dict[str, Any]:
        """Fetch a Confluence page with storage body + metadata."""
        response = self._client.get(
            f"/wiki/rest/api/content/{page_id}",
            params={
                "expand": "body.storage,version,space,history.lastUpdated,metadata.labels",
            },
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    @staticmethod
    def page_metadata(page_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extract useful metadata fields for downstream retrieval."""
        space = page_payload.get("space", {})
        version = page_payload.get("version", {})
        history = page_payload.get("history", {})
        last_updated = history.get("lastUpdated", {})
        label_results = (
            page_payload.get("metadata", {})
            .get("labels", {})
            .get("results", [])
        )
        labels = [label.get("name") for label in label_results if label.get("name")]
        return {
            "page_id": page_payload.get("id"),
            "title": page_payload.get("title"),
            "space_key": space.get("key"),
            "space_name": space.get("name"),
            "version": version.get("number"),
            "last_updated_by": last_updated.get("displayName"),
            "last_updated_on": last_updated.get("when"),
            "status": page_payload.get("status"),
            "url": ConfluenceClient.build_page_url(page_payload.get("_links", {})),
            "labels": labels,
        }

    @staticmethod
    def build_page_url(links: Dict[str, Any]) -> Optional[str]:
        """Construct an absolute page URL from the `_links` block."""
        base = links.get("base")
        webui = links.get("webui")
        if base and webui:
            return f"{base}{webui}"
        return None
