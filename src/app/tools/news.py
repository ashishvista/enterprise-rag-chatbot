"""NatWest-focused news tool that fabricates topical headlines."""
from __future__ import annotations

import datetime as _dt
import random

from langchain_core.tools import tool
from pydantic import BaseModel, Field

_CATEGORY_TEMPLATES = {
    "Top Headlines": [
        "NatWest underscores its customer-first strategy with a new service pledge in {place}.",
        "NatWest teams across {place} spotlight innovation sprints aimed at faster lending decisions.",
        "NatWest expands colleague wellbeing programmes with additional mental health support in {place}.",
    ],
    "Corporate News": [
        "NatWest Group finalises a strategic partnership with a local fintech collective in {place}.",
        "NatWest announces leadership town halls in {place} to share the bank's 2025 priorities.",
        "NatWest strengthens sustainability governance, rolling out green banking playbooks in {place}.",
    ],
    "Finance News": [
        "NatWest economists brief clients in {place} on the latest rate outlook and SME credit demand.",
        "NatWest treasury teams in {place} highlight resilient capital ratios amid shifting market sentiment.",
        "NatWest wealth specialists in {place} introduce tailored advisory bundles for high-growth sectors.",
    ],
    "Share Market News": [
        "NatWest share price sees steady trading as analysts in {place} reiterate neutral guidance.",
        "NatWest equity desk in {place} reports increased institutional interest following quarterly results.",
        "NatWest investor relations leads a webcast from {place} addressing dividend policy and capital returns.",
    ],
    "General News": [
        "NatWest volunteers in {place} partner with community groups on digital skills workshops.",
        "NatWest sustainability ambassadors in {place} launch an employee-led biodiversity challenge.",
        "NatWest innovation hub in {place} mentors startups focusing on inclusive financial tools.",
    ],
}


class GetNewsInput(BaseModel):
    """Schema for NatWest news requests."""

    location: str = Field(
        ...,
        description="City, region, or market to tailor the NatWest news digest to.",
    )


@tool(
    "get_news",
    args_schema=GetNewsInput,
    description=(
        "Return a NatWest-only news digest covering Top Headlines, Corporate News, Finance News, "
        "Share Market News, and General News for the requested location. Required argument: "
        "location (string)."
    ),
)
def get_news(location: str) -> str:
    """Return a NatWest-only news digest covering multiple categories for the given location."""

    place_raw = (location or "").strip()
    if not place_raw:
        place_raw = "NatWest's core markets"

    place_display = place_raw.title()
    date_str = _dt.datetime.utcnow().strftime("%Y-%m-%d")

    sections = []
    for category in [
        "Top Headlines",
        "Corporate News",
        "Finance News",
        "Share Market News",
        "General News",
    ]:
        templates = _CATEGORY_TEMPLATES[category]
        headline = random.choice(templates).format(place=place_display)
        sections.append(f"{category}:\n- {headline}")

    sections_text = "\n\n".join(sections)
    return (
        f"NatWest-focused updates for {place_display} on {date_str}:\n\n"
        f"{sections_text}"
    )
