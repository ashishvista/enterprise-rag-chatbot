"""Toy news tool that fabricates lighthearted headlines."""
from __future__ import annotations

import datetime as _dt
import random

from langchain_core.tools import tool

_HEADLINE_TEMPLATES = [
    "{place} startups unveil quirky AI side projects",
    "Community leaders in {place} rally for greener office rooftops",
    "Coffee aficionados declare {place} the new remote-work capital",
    "Transit pilots keep {place} commuters smiling",
    "{place} engineers host pop-up hackathon on sustainability",
    "Local artists in {place} blend murals with augmented reality",
]


@tool("get_news")
def get_news(location: str) -> str:
    """Return a few playful, randomly generated headlines for the location."""

    place_raw = location.strip()
    if not place_raw:
        place_raw = "your city"

    place_display = place_raw.title()
    date_str = _dt.datetime.utcnow().strftime("%Y-%m-%d")
    picks = random.sample(_HEADLINE_TEMPLATES, k=3)
    headlines = [headline.format(place=place_display) for headline in picks]
    formatted = "\n".join(f"- {headline}" for headline in headlines)
    return f"Top news for {place_display} on {date_str}:\n{formatted}"
