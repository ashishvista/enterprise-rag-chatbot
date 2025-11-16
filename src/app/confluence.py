"""Confluence REST helpers."""
from __future__ import annotations

from typing import Any, Dict, Optional

import httpx
from bs4 import BeautifulSoup

from .config import Settings


class ConfluenceClient:
    """Minimal Confluence Cloud REST client."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = httpx.Client(
            base_url=settings.confluence_base_url.rstrip("/"),
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
            params={"expand": "body.storage,version,space,history.lastUpdated"},
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def page_as_text(page_payload: Dict[str, Any]) -> str:
        """Convert Confluence storage HTML to readable text."""
        body_html = page_payload.get("body", {}).get("storage", {}).get("value", "")
        soup = BeautifulSoup(body_html, "html.parser")
        # Replace block elements with newlines to keep structure readable.
        for br in soup.find_all(["br", "p", "li", "h1", "h2", "h3", "h4", "h5", "h6"]):
            br.insert_after("\n")
        text = soup.get_text(separator=" ")
        return "\n".join(line.strip() for line in text.splitlines() if line.strip())

    @staticmethod
    def page_metadata(page_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extract useful metadata fields for downstream retrieval."""
        space = page_payload.get("space", {})
        version = page_payload.get("version", {})
        history = page_payload.get("history", {})
        last_updated = history.get("lastUpdated", {})
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
        }

    @staticmethod
    def build_page_url(links: Dict[str, Any]) -> Optional[str]:
        """Construct an absolute page URL from the `_links` block."""
        base = links.get("base")
        webui = links.get("webui")
        if base and webui:
            return f"{base}{webui}"
        return None
