from __future__ import annotations

"""Utilities for evaluating reasoning paths and selecting actions."""

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple


@dataclass
class ActionPlan:
    """Represents a candidate action to evaluate."""

    action: str
    utility: float
    cost: float
    rationale: str = ""


class DecisionEngine:
    """Evaluate multiple reasoning paths and select the optimal action."""

    def __init__(self, scoring_fn: Callable[[float, float], float] | None = None):
        self.scoring_fn = scoring_fn or (lambda utility, cost: utility - cost)

    def select_optimal_action(self, plans: Iterable[ActionPlan]) -> Tuple[str, str]:
        """Return the best action and its rationale among ``plans``.

        The default scoring function maximizes ``utility - cost``. The ``rationale``
        of the chosen plan is returned alongside the action. If a plan does not
        provide a rationale, a generic explanation including the computed score is
        produced.
        """

        best_plan: ActionPlan | None = None
        best_score = float("-inf")
        for plan in plans:
            score = self.scoring_fn(plan.utility, plan.cost)
            if score > best_score:
                best_score = score
                best_plan = plan
        if best_plan is None:
            raise ValueError("No plans provided")
        rationale = best_plan.rationale or (
            f"Score={best_score:.3f} (utility={best_plan.utility}, cost={best_plan.cost})"
        )
        return best_plan.action, rationale
