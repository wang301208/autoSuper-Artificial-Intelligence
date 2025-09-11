from __future__ import annotations

"""Solvers that mix symbolic rules with probabilistic reasoning."""

from typing import Dict, Iterable, List, Tuple


class RuleProbabilisticSolver:
    """Apply rule-based inference with associated probabilities."""

    def __init__(self, rules: Dict[str, List[Tuple[str, float]]]):
        """Initialize with mapping of antecedents to (conclusion, probability) tuples."""
        self.rules = rules

    def infer(self, statement: str, evidence: Iterable[str]) -> Tuple[str, float]:
        """Return the most probable conclusion given ``statement`` and ``evidence``."""
        best: Tuple[str, float] | None = None
        for antecedent, conclusions in self.rules.items():
            if antecedent == statement or antecedent in evidence:
                for conclusion, prob in conclusions:
                    if not best or prob > best[1]:
                        best = (conclusion, prob)
        return best if best else (statement, 1.0)
