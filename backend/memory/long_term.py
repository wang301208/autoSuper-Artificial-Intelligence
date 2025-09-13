"""SQLite backed long-term memory store.

The implementation is intentionally minimal – it creates a single table to
persist pieces of information along with a category.  This allows the system to
record dialogue snippets, tasks, or references to external knowledge sources
and retrieve them later.
"""

from __future__ import annotations

import sqlite3
import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence


class LongTermMemory:
    """Light‑weight long-term memory based on SQLite."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_entries: Optional[int] = None,
        vacuum_interval: int = 1000,
        cache_pages: int = 1000,
        forget_interval: int = 0,
        recycle_interval: int = 0,
    ) -> None:
        """Parameters

        path:
            Location of the sqlite database.
        max_entries:
            Optional upper bound for stored memories. When exceeded the oldest
            entries are purged.  ``None`` disables the limit.
        vacuum_interval:
            Perform a ``VACUUM`` after this many insert operations. ``0``
            disables automatic vacuuming.
        forget_interval:
            Apply the forgetting strategy after this many insert operations.
            ``0`` disables automatic forgetting.
        recycle_interval:
            Apply the recycling strategy after this many insert operations.
            ``0`` disables automatic recycling.
        cache_pages:
            Page count for the SQLite cache.  Smaller values keep the memory
            footprint low.
        """

        self.path = Path(path)
        self.conn = sqlite3.connect(self.path)

        # Tune the database for a small memory footprint and concurrent access
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(f"PRAGMA cache_size={int(cache_pages)}")

        self.max_entries = max_entries
        self.vacuum_interval = vacuum_interval
        self.forget_interval = forget_interval
        self.recycle_interval = recycle_interval
        self.forget_strategy: Optional[Callable[[sqlite3.Connection], None]] = None
        self.recycle_strategy: Optional[Callable[[sqlite3.Connection], None]] = None
        self.compression_strategy: Optional[Callable[[sqlite3.Connection], None]] = None
        self._pending_adds = 0
        self._total_adds = 0
        self._in_batch = False

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
        # Table for embedding vectors
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                key TEXT PRIMARY KEY,
                vector TEXT NOT NULL,
                metadata TEXT
            )
            """
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Strategy configuration
    # ------------------------------------------------------------------
    def set_compression_strategy(
        self, strategy: Callable[[sqlite3.Connection], None]
    ) -> None:
        """Define custom compression strategy."""

        self.compression_strategy = strategy

    def set_forget_strategy(self, strategy: Callable[[sqlite3.Connection], None]) -> None:
        """Define custom forgetting strategy."""

        self.forget_strategy = strategy

    def set_recycle_strategy(
        self, strategy: Callable[[sqlite3.Connection], None]
    ) -> None:
        """Define custom recycling strategy."""

        self.recycle_strategy = strategy

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
        self._pending_adds += 1
        if not self._in_batch:
            self.conn.commit()
            self._post_commit()

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

    # ------------------------------------------------------------------
    # Embedding specific helpers
    # ------------------------------------------------------------------
    def add_embedding(
        self, key: str, vector: Sequence[float], metadata: Optional[dict] = None
    ) -> None:
        """Store an embedding vector and optional metadata."""

        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO embeddings (key, vector, metadata) VALUES (?, ?, ?)",
            (
                key,
                json.dumps(list(map(float, vector))),
                json.dumps(metadata) if metadata is not None else None,
            ),
        )
        self.conn.commit()

    def get_embedding(self, key: str) -> Optional[tuple[list[float], Optional[dict]]]:
        """Retrieve an embedding and metadata for ``key``."""

        cur = self.conn.cursor()
        cur.execute("SELECT vector, metadata FROM embeddings WHERE key = ?", (key,))
        row = cur.fetchone()
        if row is None:
            return None
        vector = json.loads(row[0])
        metadata = json.loads(row[1]) if row[1] is not None else None
        return vector, metadata

    def iter_embeddings(self) -> Iterable[tuple[str, list[float], Optional[dict]]]:
        """Iterate over all stored embeddings."""

        cur = self.conn.cursor()
        cur.execute("SELECT key, vector, metadata FROM embeddings")
        for key, vec, meta in cur.fetchall():
            yield key, json.loads(vec), json.loads(meta) if meta is not None else None

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------
    # Transaction helpers and maintenance
    # ------------------------------------------------------------------
    @contextmanager
    def batch(self):
        """Group multiple ``add`` calls into a single transaction."""

        self._in_batch = True
        try:
            yield self
            self.conn.commit()
            self._post_commit()
        except Exception:  # pragma: no cover - rollback path
            self.conn.rollback()
            self._pending_adds = 0
            raise
        finally:
            self._in_batch = False

    def _post_commit(self) -> None:
        """Run housekeeping tasks after committing new entries."""

        self._total_adds += self._pending_adds
        self._pending_adds = 0

        if self.max_entries is not None:
            self._trim_to_limit()

        if (
            self.forget_interval
            and self.forget_strategy
            and self._total_adds % self.forget_interval == 0
        ):
            self.forget()

        if (
            self.recycle_interval
            and self.recycle_strategy
            and self._total_adds % self.recycle_interval == 0
        ):
            self.recycle()

        if self.vacuum_interval and self._total_adds % self.vacuum_interval == 0:
            self.compress()

    def compress(
        self, strategy: Optional[Callable[[sqlite3.Connection], None]] = None
    ) -> None:
        """Apply compression strategy or fall back to ``VACUUM``."""

        strategy = strategy or self.compression_strategy
        if strategy is not None:
            strategy(self.conn)
            self.conn.commit()
        else:
            self.vacuum()

    def forget(
        self,
        *,
        before_ts: Optional[float] = None,
        strategy: Optional[Callable[[sqlite3.Connection], None]] = None,
    ) -> None:
        """Forget memories using ``strategy`` or timestamp filtering."""

        strategy = strategy or self.forget_strategy
        if strategy is not None:
            strategy(self.conn)
            self.conn.commit()
            return
        if before_ts is None:
            return
        cur = self.conn.cursor()
        cur.execute("DELETE FROM memory WHERE timestamp < ?", (before_ts,))
        self.conn.commit()

    def recycle(
        self, strategy: Optional[Callable[[sqlite3.Connection], None]] = None
    ) -> None:
        """Recycle memory entries using ``strategy`` or ``VACUUM``."""

        strategy = strategy or self.recycle_strategy
        if strategy is not None:
            strategy(self.conn)
            self.conn.commit()
        else:
            self.vacuum()

    def _trim_to_limit(self) -> None:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM memory")
        (count,) = cur.fetchone()
        if count <= self.max_entries:
            return
        remove = count - self.max_entries
        cur.execute(
            "DELETE FROM memory WHERE id IN ("
            "SELECT id FROM memory ORDER BY timestamp ASC LIMIT ?"
            ")",
            (remove,),
        )
        self.conn.commit()

    def vacuum(self) -> None:
        """Reclaim unused space in the database."""

        cur = self.conn.cursor()
        cur.execute("VACUUM")
        self.conn.commit()

