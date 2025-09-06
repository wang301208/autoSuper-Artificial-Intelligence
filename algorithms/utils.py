"""Common data structures and helper functions for algorithms."""
from typing import Any, Dict, List


def swap(items: List[Any], i: int, j: int) -> None:
    """Swap two elements in a list in-place.

    Time complexity: O(1)
    Space complexity: O(1)
    """
    items[i], items[j] = items[j], items[i]


class Graph:
    """Simple undirected graph using an adjacency list."""

    def __init__(self) -> None:
        self.adjacency: Dict[Any, List[Any]] = {}

    def add_edge(self, u: Any, v: Any) -> None:
        """Add an undirected edge between ``u`` and ``v``.

        Time complexity: O(1)
        Space complexity: O(1)
        """
        self.adjacency.setdefault(u, []).append(v)
        self.adjacency.setdefault(v, []).append(u)

    def neighbors(self, node: Any) -> List[Any]:
        """Return neighbors of ``node``."""
        return self.adjacency.get(node, [])
