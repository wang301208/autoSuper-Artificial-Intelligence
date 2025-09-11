from __future__ import annotations

"""Planning utilities for chaining reasoning steps."""

from typing import Dict, Iterable, List, Tuple

from .interfaces import KnowledgeSource, Solver
from .decision_engine import ActionPlan, DecisionEngine


class ReasoningPlanner:
    """Plan and execute reasoning steps with caching and explanations."""

    def __init__(
        self,
        knowledge_sources: List[KnowledgeSource] | None = None,
        solver: Solver | None = None,
        decision_engine: DecisionEngine | None = None,
    ):
        self.knowledge_sources = knowledge_sources or []
        self.solver = solver
        self.decision_engine = decision_engine
        self.cache: Dict[str, Tuple[str, float]] = {}
        self.history: List[Dict[str, object]] = []

    def infer(self, statement: str) -> Tuple[str, float]:
        """Infer a conclusion for ``statement`` leveraging all knowledge sources."""
        if statement in self.cache:
            return self.cache[statement]
        evidence: List[str] = []
        for source in self.knowledge_sources:
            evidence.extend(source.query(statement))
        if self.solver:
            conclusion, probability = self.solver.infer(statement, evidence)
        else:
            conclusion, probability = statement, 1.0
        self.cache[statement] = (conclusion, probability)
        self.history.append(
            {
                "statement": statement,
                "conclusion": conclusion,
                "probability": probability,
                "evidence": evidence,
            }
        )
        return conclusion, probability

    def chain(self, statements: Iterable[str]) -> List[Tuple[str, float]]:
        """Sequentially infer conclusions for multiple statements."""
        return [self.infer(s) for s in statements]

    def explain(self) -> List[Dict[str, object]]:
        """Return a trace of all reasoning steps performed so far."""
        return list(self.history)

    def plan(self, options: Iterable[Dict[str, float | str]]) -> Tuple[str, str]:
        """Select the optimal action among ``options`` using the decision engine.

        Each option should provide an ``action`` string and ``cost``. If ``utility``
        is omitted, it is estimated via :meth:`infer` on the action string. The
        chosen action and its rationale are returned.
        """

        if not self.decision_engine:
            raise ValueError("No decision engine available")

        plans: List[ActionPlan] = []
        for option in options:
            action = str(option["action"])
            cost = float(option.get("cost", 0))
            if "utility" in option:
                utility = float(option["utility"])
                rationale = str(option.get("rationale", ""))
            else:
                conclusion, probability = self.infer(action)
                utility = probability
                rationale = str(option.get("rationale", conclusion))
            plans.append(ActionPlan(action=action, utility=utility, cost=cost, rationale=rationale))

        return self.decision_engine.select_optimal_action(plans)
