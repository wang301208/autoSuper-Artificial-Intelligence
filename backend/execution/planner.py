"""Simple planner to decompose high level goals into executable sub-tasks."""
from __future__ import annotations

from typing import List

from backend.creative_engine.problem_solver import DivergentConvergentSolver


class Planner:
    """Decompose high level goals into ordered sub-tasks."""

    def __init__(self, solver: DivergentConvergentSolver | None = None) -> None:
        self.solver = solver

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

    def solve(self, goal: str | dict) -> List[str]:
        """Return a plan for ``goal``.

        When ``goal`` specifies multiple strategies via a mapping with
        ``start``, ``goal`` and ``strategies`` keys, the
        :class:`DivergentConvergentSolver` is used to pick the best path.
        Otherwise the goal string is simply decomposed into sub-tasks.
        """

        if isinstance(goal, dict) and goal.get("strategies"):
            if not self.solver:
                raise ValueError("No solver configured")
            start = goal.get("start", "")
            target = goal.get("goal", "")
            strategies = goal.get("strategies", [])
            best_path, _ = self.solver.solve(start, target, strategies)
            return best_path
        return self.decompose(goal if isinstance(goal, str) else "")
