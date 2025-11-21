"""Postgres-backed conversation history persistence."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

from psycopg import sql

from ..config import Settings, async_db_connection


@dataclass(frozen=True)
class ConversationMessage:
    """Single conversational turn stored in the database."""

    session_id: str
    role: str
    content: str
    created_at: datetime
    message_index: int


class ConversationHistoryStore:
    """Persist and retrieve conversation messages for chat sessions."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._table_ready = False
        self._lock = asyncio.Lock()

    @property
    def table_name(self) -> str:
        return self._settings.conversation_history_table

    @property
    def schema_name(self) -> str:
        return self._settings.database_schema

    async def ensure_table(self) -> None:
        if self._table_ready:
            return
        async with self._lock:
            if self._table_ready:
                return
            await self._create_table()
            self._table_ready = True

    async def _create_table(self) -> None:
        qualified_table = sql.Identifier(self.schema_name, self.table_name)
        unique_index = sql.Identifier(f"{self.table_name}_session_idx")
        async with async_db_connection(self._settings) as conn:
            async with conn.cursor() as cur:
                create_stmt = sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {table} (
                        id BIGSERIAL PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        message_index INTEGER NOT NULL,
                        UNIQUE(session_id, message_index)
                    )
                    """
                ).format(table=qualified_table)
                await cur.execute(create_stmt)

                index_stmt = sql.SQL(
                    """
                    CREATE INDEX IF NOT EXISTS {index_name}
                    ON {table} (session_id, message_index)
                    """
                ).format(index_name=unique_index, table=qualified_table)
                await cur.execute(index_stmt)
            await conn.commit()

    async def add_message(self, session_id: str, role: str, content: str) -> ConversationMessage:
        await self.ensure_table()
        qualified_table = sql.Identifier(self.schema_name, self.table_name)
        insert_stmt = sql.SQL(
            """
            INSERT INTO {table} (session_id, role, content, message_index)
            VALUES (
                %s,
                %s,
                %s,
                COALESCE((SELECT MAX(message_index) + 1 FROM {table} WHERE session_id = %s), 0)
            )
            RETURNING session_id, role, content, created_at, message_index
            """
        ).format(table=qualified_table)
        async with async_db_connection(self._settings) as conn:
            async with conn.cursor() as cur:
                await cur.execute(insert_stmt, (session_id, role, content, session_id))
                row = await cur.fetchone()
            await conn.commit()
        assert row is not None
        return ConversationMessage(
            session_id=row[0],
            role=row[1],
            content=row[2],
            created_at=row[3],
            message_index=row[4],
        )

    async def add_messages(self, session_id: str, messages: Sequence[tuple[str, str]]) -> List[ConversationMessage]:
        results: List[ConversationMessage] = []
        for role, content in messages:
            results.append(await self.add_message(session_id, role, content))
        return results

    async def fetch_recent_messages(self, session_id: str, limit: Optional[int] = None) -> List[ConversationMessage]:
        await self.ensure_table()
        qualified_table = sql.Identifier(self.schema_name, self.table_name)
        base_query = sql.SQL(
            """
            SELECT session_id, role, content, created_at, message_index
            FROM {table}
            WHERE session_id = %s
            ORDER BY message_index DESC
            """
        ).format(table=qualified_table)
        if limit is not None:
            query = base_query + sql.SQL(" LIMIT %s")
            params: Tuple[object, ...] = (session_id, limit)
        else:
            query = base_query
            params = (session_id,)
        async with async_db_connection(self._settings) as conn:
            async with conn.cursor() as cur:
                await cur.execute(query, params)
                rows = await cur.fetchall()
        messages = [
            ConversationMessage(
                session_id=row[0],
                role=row[1],
                content=row[2],
                created_at=row[3],
                message_index=row[4],
            )
            for row in rows
        ]
        messages.reverse()
        return messages

    async def delete_session(self, session_id: str) -> None:
        await self.ensure_table()
        qualified_table = sql.Identifier(self.schema_name, self.table_name)
        delete_stmt = sql.SQL("DELETE FROM {table} WHERE session_id = %s").format(table=qualified_table)
        async with async_db_connection(self._settings) as conn:
            async with conn.cursor() as cur:
                await cur.execute(delete_stmt, (session_id,))
            await conn.commit()
