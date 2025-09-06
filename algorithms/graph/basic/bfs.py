"""Breadth-first search algorithm."""
from collections import deque
from ...base import Algorithm
from ...utils import Graph
from typing import Any, List


class BreadthFirstSearch(Algorithm):
    """Traverse a graph using BFS."""

    def execute(self, graph: Graph, start: Any) -> List[Any]:
        """Return nodes in BFS order starting from ``start``.

        Time complexity: O(V + E)
        Space complexity: O(V)
        """
        visited: List[Any] = []
        queue: deque[Any] = deque([start])
        seen = {start}
        while queue:
            node = queue.popleft()
            visited.append(node)
            for neighbor in graph.neighbors(node):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        return visited
