"""Tool registry for the chatbot."""
from __future__ import annotations

from typing import Sequence

from langchain_core.tools import BaseTool

from .knowledge_base import query_natwest_knowledge_base
from .news import get_news
from .slx_requests import get_slx_request_status, raise_slx_request
from .speak_up import (
    get_speak_up_status,
    raise_speak_up_complaint,
    withdraw_speak_up_complaint,
)


def get_default_tools() -> Sequence[BaseTool]:
    """Return the default set of tools available to the chatbot."""

    return [
        get_news,
        query_natwest_knowledge_base,
        raise_speak_up_complaint,
        get_speak_up_status,
        withdraw_speak_up_complaint,
        raise_slx_request,
        get_slx_request_status,
    ]
