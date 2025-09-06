"""Depth-first search algorithm."""
from ...base import Algorithm
from ...utils import Graph
from typing import Any, List, Set


class DepthFirstSearch(Algorithm):
    """Traverse a graph using DFS."""

    def execute(self, graph: Graph, start: Any) -> List[Any]:
        """Return nodes in DFS order starting from ``start``.

        Time complexity: O(V + E)
        Space complexity: O(V)
        """
        visited: List[Any] = []
        self._dfs(graph, start, visited, set())
        return visited

    def _dfs(self, graph: Graph, node: Any, visited: List[Any], seen: Set[Any]) -> None:
        seen.add(node)
        visited.append(node)
        for neighbor in graph.neighbors(node):
            if neighbor not in seen:
                self._dfs(graph, neighbor, visited, seen)
