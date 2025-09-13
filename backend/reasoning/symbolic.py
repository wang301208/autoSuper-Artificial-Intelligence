from __future__ import annotations

"""Symbolic reasoning with chain-of-thought generation."""

from typing import Iterable, List, Tuple

from .decision_engine import ActionPlan, DecisionEngine
from .multi_hop import MultiHopAssociator
from .causal import KnowledgeGraphCausalReasoner, CounterfactualGraphReasoner


class SymbolicReasoner:
    """Perform symbolic reasoning over a knowledge graph."""

    def __init__(
        self,
        graph: dict[str, Iterable[str]],
        decision_engine: DecisionEngine | None = None,
    ):
        self.graph = graph
        self.associator = MultiHopAssociator(graph)
        self.decision_engine = decision_engine or DecisionEngine()
        self.causal = KnowledgeGraphCausalReasoner(graph)
        self.counterfactual = CounterfactualGraphReasoner(self.causal)

    def chain_of_thought(self, start: str, goal: str) -> List[str]:
        """Generate intermediate reasoning steps from ``start`` to ``goal``."""

        path = self.associator.find_path(start, goal)
        return [f"{path[i]} -> {path[i + 1]}" for i in range(len(path) - 1)]

    def reason(self, start: str, goal: str) -> Tuple[str, List[str]]:
        """Return the conclusion and chain-of-thought."""

        steps = self.chain_of_thought(start, goal)
        conclusion = goal if steps else start
        if steps:
            plans = [ActionPlan(action=s, utility=1.0, cost=0.0, rationale=s) for s in steps]
            # Decision engine evaluates the steps; we ignore the choice but keep integration
            self.decision_engine.select_optimal_action(plans)
        return conclusion, steps

    def explain_causality(self, cause: str, effect: str) -> Tuple[bool, Iterable[str]]:
        """Expose causal links between ``cause`` and ``effect``."""

        return self.causal.check_causality(cause, effect)

    def evaluate_counterfactual(self, cause: str, effect: str) -> str:
        """Provide a counterfactual explanation for ``effect`` if ``cause`` changes."""

        return self.counterfactual.evaluate_counterfactual(cause, effect)

