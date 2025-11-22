"""Database connection helpers."""
from __future__ import annotations

import os
import socket
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional, Any

import psycopg
from psycopg import AsyncConnection
from sqlalchemy.engine.url import URL, make_url

from .settings import Settings

DEFAULT_SERVICE_HOSTS = {"pgvector-db", "postgres"}


def _url_to_string(url: URL) -> str:
    """Render a SQLAlchemy URL while preserving secrets like passwords."""
    return url.render_as_string(hide_password=False)


def to_sync_sqlalchemy_url(base_url: str) -> str:
    """Convert the base Postgres URL to a SQLAlchemy sync driver URL."""
    url = make_url(base_url)
    sync_url = _url_to_string(url.set(drivername="postgresql+psycopg"))
    return sync_url


def to_async_sqlalchemy_url(base_url: str) -> str:
    """Convert the base Postgres URL to a SQLAlchemy async driver URL."""
    url = make_url(base_url)
    async_url = _url_to_string(url.set(drivername="postgresql+asyncpg"))
    return async_url


def to_psycopg_dsn(base_url: str) -> str:
    """Produce a psycopg-compatible DSN from the base Postgres URL."""
    url = make_url(base_url)
    dsn = _url_to_string(url.set(drivername="postgresql"))
    return dsn


async def get_async_connection(settings: Settings) -> AsyncConnection:
    """Create a new psycopg async connection using settings."""
    dsn = to_psycopg_dsn(settings.base_db_url())
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
