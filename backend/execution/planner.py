"""Simple planner to decompose high level goals into executable sub-tasks."""
from __future__ import annotations

from typing import List


class Planner:
    """Decompose high level goals into ordered sub-tasks."""

    def decompose(self, goal: str, source: str | None = None) -> List[str]:
        """Return a list of sub-tasks derived from a high level goal.

        Parameters
        ----------
        goal:
            The objective to break down into smaller tasks.
        source:
            Optional tag describing where the goal originated.  When
            provided, each resulting task is annotated to retain this
            provenance information.

        The default implementation uses a few heuristic separators to
        break the goal into manageable pieces. It can be replaced with a
        more sophisticated planner or LLM powered approach in the future.
        """
        if not goal:
            return []
        # Replace common separators with newlines to unify splitting logic
        separators = ["\n", ";", " and ", " then "]
        normalized = goal
        for sep in separators[1:]:
            normalized = normalized.replace(sep, "\n")
        tasks = [task.strip() for task in normalized.splitlines() if task.strip()]
        if source:
            tasks = [f"{task} [{source}]" for task in tasks]
        return tasks
