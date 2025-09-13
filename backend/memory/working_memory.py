"""Short-term working memory implementation."""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, Any


class WorkingMemory:
    """FIFO working memory with a fixed capacity."""

    def __init__(self, capacity: int = 10) -> None:
        self.capacity = capacity
        self._items: Deque[Any] = deque(maxlen=capacity)

    def add(self, item: Any) -> None:
        """Add ``item`` to the working memory."""
        self._items.append(item)

    def get(self) -> list[Any]:
        """Return current items in order of insertion."""
        return list(self._items)

    def clear(self) -> None:
        """Remove all items from working memory."""
        self._items.clear()
