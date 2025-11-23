"""Tool registry for the chatbot."""
from __future__ import annotations

from typing import Sequence

from langchain_core.tools import BaseTool

from .news import get_news
from .weather import get_weather
from .knowledge_base import query_natwest_knowledge_base


def get_default_tools() -> Sequence[BaseTool]:
    """Return the default set of tools available to the chatbot."""

    return [get_weather, get_news, query_natwest_knowledge_base]
