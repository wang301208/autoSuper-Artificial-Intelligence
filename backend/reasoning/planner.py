from __future__ import annotations

"""Planning utilities for chaining reasoning steps."""

from typing import Dict, Iterable, List, Tuple

from .interfaces import KnowledgeSource, Solver


class ReasoningPlanner:
    """Plan and execute reasoning steps with caching and explanations."""

    def __init__(
        self,
        knowledge_sources: List[KnowledgeSource] | None = None,
        solver: Solver | None = None,
    ):
        self.knowledge_sources = knowledge_sources or []
        self.solver = solver
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
