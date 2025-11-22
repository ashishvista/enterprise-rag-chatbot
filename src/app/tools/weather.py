"""Lightweight weather tool that returns a playful random report."""
from __future__ import annotations

import random

from langchain_core.tools import tool

_CONDITIONS = [
    "sunny with a gentle breeze",
    "mostly cloudy but comfortable",
    "light rain and a cozy chill",
    "gusty winds and dramatic clouds",
    "clear skies perfect for a walk",
    "scattered showers that pass quickly",
]


@tool("get_weather")
def get_weather(location: str) -> str:
    """Return a whimsical weather blurb for the supplied location."""

    place = location.strip() or "your area"
    condition = random.choice(_CONDITIONS)
    temperature = random.randint(12, 32)
    humidity = random.randint(30, 90)
    return (
        f"Forecast for {place}: {condition}. "
        f"Expect around {temperature}°C with humidity near {humidity}%"
    )
