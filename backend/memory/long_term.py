"""SQLite backed long-term memory store.

The implementation is intentionally minimal – it creates a single table to
persist pieces of information along with a category.  This allows the system to
record dialogue snippets, tasks, or references to external knowledge sources
and retrieve them later.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence


class LongTermMemory:
    """Light‑weight long-term memory based on SQLite."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path)
        self._init_db()

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL DEFAULT (strftime('%s','now')),
                tags TEXT
            )
            """
        )
        # Migration for existing databases missing new columns
        cur.execute("PRAGMA table_info(memory)")
        columns = {row[1] for row in cur.fetchall()}
        if "timestamp" not in columns:
            cur.execute(
                "ALTER TABLE memory ADD COLUMN timestamp REAL NOT NULL DEFAULT (strftime('%s','now'))"
            )
        if "tags" not in columns:
            cur.execute("ALTER TABLE memory ADD COLUMN tags TEXT")
        # Create indexes for faster lookups
        cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_category ON memory(category)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_timestamp ON memory(timestamp)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_tags ON memory(tags)")
        self.conn.commit()

    def add(
        self,
        category: str,
        content: str,
        *,
        tags: Optional[Sequence[str]] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Store a piece of ``content`` under ``category``."""

        if timestamp is None:
            timestamp = datetime.utcnow().timestamp()
        tags_str = ",".join(sorted(tags)) if tags else None
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO memory (category, content, timestamp, tags) VALUES (?, ?, ?, ?)",
            (category, content, timestamp, tags_str),
        )
        self.conn.commit()

    def get(
        self,
        category: Optional[str] = None,
        *,
        tags: Optional[Sequence[str]] = None,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
    ) -> Iterable[str]:
        """Retrieve stored memories.

        Parameters
        ----------
        category:
            If provided, filter results by this category.  Otherwise all stored
            memories are returned.
        tags:
            Only return memories that include all of these tags.
        start_ts, end_ts:
            Restrict results to this time range (Unix timestamps). ``start_ts``
            is inclusive and ``end_ts`` is exclusive.
        """

        cur = self.conn.cursor()
        query = "SELECT content FROM memory"
        conditions = []
        params: list[object] = []
        if category is not None:
            conditions.append("category = ?")
            params.append(category)
        if tags:
            for tag in tags:
                conditions.append("tags LIKE ?")
                params.append(f"%{tag}%")
        if start_ts is not None:
            conditions.append("timestamp >= ?")
            params.append(start_ts)
        if end_ts is not None:
            conditions.append("timestamp < ?")
            params.append(end_ts)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        cur.execute(query, params)
        for (content,) in cur.fetchall():
            yield content

    def close(self) -> None:
        self.conn.close()
