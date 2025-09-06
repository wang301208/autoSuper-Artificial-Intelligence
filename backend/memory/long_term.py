"""SQLite backed long-term memory store.

The implementation is intentionally minimal – it creates a single table to
persist pieces of information along with a category.  This allows the system to
record dialogue snippets, tasks, or references to external knowledge sources
and retrieve them later.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Optional


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
                content TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def add(self, category: str, content: str) -> None:
        """Store a piece of ``content`` under ``category``."""

        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO memory (category, content) VALUES (?, ?)",
            (category, content),
        )
        self.conn.commit()

    def get(self, category: Optional[str] = None) -> Iterable[str]:
        """Retrieve stored memories.

        Parameters
        ----------
        category:
            If provided, filter results by this category.  Otherwise all stored
            memories are returned.
        """

        cur = self.conn.cursor()
        if category is None:
            cur.execute("SELECT content FROM memory")
        else:
            cur.execute("SELECT content FROM memory WHERE category = ?", (category,))
        for (content,) in cur.fetchall():
            yield content

    def close(self) -> None:
        self.conn.close()
