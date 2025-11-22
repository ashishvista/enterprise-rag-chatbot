"""Weather tool fetching sample data for a given location."""
from __future__ import annotations

import httpx

from langchain_core.tools import tool

_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


@tool("get_weather")
def get_weather(location: str) -> str:
    """Return a brief weather summary for the provided location (city or lat,long)."""

    if not location:
        return "Please provide a city name or latitude,longitude coordinates."

    coordinates = _maybe_parse_coordinates(location)
    if coordinates is None:
        geocode = _lookup_coordinates(location)
        if geocode is None:
            return f"Could not resolve '{location}' to coordinates."
        coordinates = geocode

    latitude, longitude = coordinates
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,relativehumidity_2m",
        "current_weather": True,
    }
    try:
        response = httpx.get(_OPEN_METEO_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # pragma: no cover - network variability
        return f"Weather service error: {exc}"

    current = data.get("current_weather", {})
    temperature = current.get("temperature")
    windspeed = current.get("windspeed")
    summary = current.get("weathercode")
    humidity = (
        (data.get("hourly", {}).get("relativehumidity_2m", []) or [None])[0]
        if data.get("hourly")
        else None
    )
    summary_parts = [
        f"Temperature: {temperature}°C" if temperature is not None else None,
        f"Humidity: {humidity}%" if humidity is not None else None,
        f"Wind: {windspeed} km/h" if windspeed is not None else None,
        f"Weather code: {summary}" if summary is not None else None,
    ]
    formatted = ", ".join(part for part in summary_parts if part)
    return formatted or "Weather data unavailable."


def _maybe_parse_coordinates(value: str) -> tuple[float, float] | None:
    try:
        lat_str, lon_str = [part.strip() for part in value.split(",", 1)]
        return float(lat_str), float(lon_str)
    except Exception:
        return None


def _lookup_coordinates(location: str) -> tuple[float, float] | None:
    url = "https://geocoding-api.open-meteo.com/v1/search"
    try:
        response = httpx.get(url, params={"name": location, "count": 1}, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception:  # pragma: no cover
        return None

    results = data.get("results") or []
    if not results:
        return None
    first = results[0]
    return float(first["latitude"]), float(first["longitude"])
