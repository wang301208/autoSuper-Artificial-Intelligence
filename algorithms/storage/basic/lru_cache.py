"""Least Recently Used (LRU) cache implementation."""
from collections import OrderedDict
from typing import Any

from ...base import Algorithm


class LRUCache(Algorithm):
    """Simple LRU cache.

    The cache discards the least recently used items first when the capacity is
    exceeded.

    Attributes
    ----------
    capacity: int
        Maximum number of items to store.

    Operations
    ----------
    get(key): Retrieve a value by key. *Time*: O(1), *Space*: O(1).
    put(key, value): Insert or update a value. *Time*: O(1), *Space*: O(1).
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.cache: "OrderedDict[Any, Any]" = OrderedDict()

    def get(self, key: Any) -> Any | None:
        """Return value for ``key`` or ``None`` if missing."""
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: Any, value: Any) -> None:
        """Insert ``value`` under ``key``.

        If the cache exceeds its capacity, the least recently used item is
        evicted.
        """
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def execute(self, *args, **kwargs) -> dict[Any, Any]:
        """Return a snapshot of the current cache state."""
        return dict(self.cache)
