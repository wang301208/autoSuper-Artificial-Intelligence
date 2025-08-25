"""Time-series storage for aggregating event bus data."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable

from events import subscribe


class TimeSeriesStorage:
    """Persist events from the global event bus into SQLite."""

    def __init__(self, db_path: Path | str = "monitoring.db") -> None:
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._init_db()

    def _init_db(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                ts REAL,
                topic TEXT,
                data TEXT
            )
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # event ingestion
    # ------------------------------------------------------------------
    def store(self, topic: str, event: Dict[str, Any]) -> None:
        """Store *event* published on *topic*."""
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO events (ts, topic, data) VALUES (?, ?, ?)",
            (time.time(), topic, json.dumps(event)),
        )
        self._conn.commit()

    def subscribe_to(self, topics: Iterable[str]) -> None:
        """Subscribe to *topics* on the global event bus and persist events."""
        for topic in topics:
            subscribe(topic, lambda e, t=topic: self.store(t, e))

    # ------------------------------------------------------------------
    # event retrieval
    # ------------------------------------------------------------------
    def events(self, topic: str | None = None) -> list[Dict[str, Any]]:
        """Return stored events, optionally filtered by *topic*."""
        cur = self._conn.cursor()
        if topic is None:
            cur.execute("SELECT data FROM events")
        else:
            cur.execute("SELECT data FROM events WHERE topic=?", (topic,))
        rows = cur.fetchall()
        return [json.loads(r[0]) for r in rows]

    # ------------------------------------------------------------------
    # aggregations
    # ------------------------------------------------------------------
    def _all_events(self) -> list[Dict[str, Any]]:
        return self.events()

    def success_rate(self) -> float:
        """Return overall success rate from stored events."""
        events = self._all_events()
        if not events:
            return 0.0
        successes = sum(1 for e in events if e.get("status") == "success")
        return successes / len(events)

    def bottlenecks(self) -> Dict[str, int]:
        """Return counts of failed events grouped by stage."""
        events = self._all_events()
        result: Dict[str, int] = {}
        for ev in events:
            if ev.get("status") != "success":
                stage = ev.get("stage", "unknown")
                result[stage] = result.get(stage, 0) + 1
        return result

    def blueprint_versions(self) -> Dict[str, int]:
        """Return counts of events grouped by blueprint version."""
        events = self._all_events()
        versions: Dict[str, int] = {}
        for ev in events:
            ver = str(ev.get("blueprint_version", "unknown"))
            versions[ver] = versions.get(ver, 0) + 1
        return versions

    # ------------------------------------------------------------------
    # cleanup
    # ------------------------------------------------------------------
    def close(self) -> None:
        self._conn.close()
