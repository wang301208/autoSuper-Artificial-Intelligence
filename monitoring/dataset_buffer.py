"""SQLite-backed dataset buffer for accumulating logs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict


class DatasetBuffer:
    """Store log entries in an SQLite database for later processing."""

    def __init__(self, db_path: Path | str = "dataset_buffer.db") -> None:
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._init_db()

    def _init_db(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data TEXT
            )
            """
        )
        self._conn.commit()

    def add_log(self, log: Dict[str, Any]) -> None:
        """Persist a *log* entry into the buffer."""
        cur = self._conn.cursor()
        cur.execute("INSERT INTO logs (data) VALUES (?)", (json.dumps(log),))
        self._conn.commit()

    def fetch_logs(self) -> list[Dict[str, Any]]:
        """Return all stored log entries."""
        cur = self._conn.cursor()
        cur.execute("SELECT data FROM logs ORDER BY id")
        rows = cur.fetchall()
        return [json.loads(r[0]) for r in rows]

    def clear(self) -> None:
        """Remove all log entries from the buffer."""
        cur = self._conn.cursor()
        cur.execute("DELETE FROM logs")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
