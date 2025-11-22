"""News tool returning recent headlines for a location."""
from __future__ import annotations

import datetime as _dt
from typing import List

from langchain_core.tools import tool

# Placeholder in-memory feed for demonstration purposes.
_SAMPLE_HEADLINES = {
    "new york": [
        "Local Tech Conference Highlights AI Safety Initiatives",
        "City Council Advances Sustainable Transit Plan",
        "Community Gardens Expand Across Boroughs",
    ],
    "san francisco": [
        "Bay Area Startups Secure Record Funding",
        "Golden Gate Bridge Maintenance Scheduled Overnight",
        "Local Artists Lead Waterfront Revitalization",
    ],
}


@tool("get_news")
def get_news(location: str) -> str:
    """Return a short list of recent headlines mentioning the location."""

    if not location:
        return "Provide a city or region to look up related headlines."

    normalized = location.lower().strip()
    headlines: List[str] | None = _SAMPLE_HEADLINES.get(normalized)
    if headlines is None:
        headlines = [
            f"No curated headlines for '{location}'. Check major outlets for the latest updates.",
        ]

    date_str = _dt.datetime.utcnow().strftime("%Y-%m-%d")
    formatted = "\n".join(f"- {headline}" for headline in headlines)
    return f"Top news for {location} on {date_str}:\n{formatted}"
