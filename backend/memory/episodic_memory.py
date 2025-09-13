"""Episodic memory storing time stamped experiences."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional


@dataclass
class Episode:
    timestamp: float
    data: str


class EpisodicMemory:
    """Simple list based episodic memory."""

    def __init__(self) -> None:
        self._episodes: List[Episode] = []

    def add(self, data: str, *, timestamp: Optional[float] = None) -> None:
        """Record a new episode."""
        if timestamp is None:
            timestamp = datetime.utcnow().timestamp()
        self._episodes.append(Episode(timestamp, data))

    def get(
        self,
        *,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
    ) -> Iterable[str]:
        """Yield episodes within the provided time window."""
        for ep in self._episodes:
            if start_ts is not None and ep.timestamp < start_ts:
                continue
            if end_ts is not None and ep.timestamp >= end_ts:
                continue
            yield ep.data

    def clear(self) -> None:
        """Remove all stored episodes."""
        self._episodes.clear()
