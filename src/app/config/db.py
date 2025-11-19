"""Database connection helpers."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional, Any

import psycopg
from psycopg import AsyncConnection

from .settings import Settings


def _strip_sqlalchemy_driver(dsn: str) -> str:
    """Remove SQLAlchemy driver specifiers so psycopg can consume the DSN."""
    cleaned = dsn
    for marker in ("+asyncpg", "+psycopg"):
        if marker in cleaned:
            cleaned = cleaned.replace(marker, "", 1)
    return cleaned


async def get_async_connection(settings: Settings) -> AsyncConnection:
    """Create a new psycopg async connection using settings."""
    dsn = _strip_sqlalchemy_driver(settings.async_db_url())
    return await psycopg.AsyncConnection.connect(dsn)


@asynccontextmanager
async def async_db_connection(settings: Settings) -> AsyncIterator[AsyncConnection]:
    """Yield an async psycopg connection and close it afterwards."""
    conn = await get_async_connection(settings)
    try:
        yield conn
    finally:
        await conn.close()


async def fetch_scalar(query: str, settings: Settings) -> Optional[Any]:
    """Execute a query and return the first column of the first row."""
    async with async_db_connection(settings) as conn:
        async with conn.cursor() as cur:
            await cur.execute(query)
            row = await cur.fetchone()
    if not row:
        return None
    return row[0]
