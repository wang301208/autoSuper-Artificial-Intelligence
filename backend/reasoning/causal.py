from __future__ import annotations

"""Causal and counterfactual reasoning helpers."""

from typing import Iterable, Tuple

from .multi_hop import MultiHopAssociator
from .interfaces import CausalReasoner, CounterfactualReasoner


class KnowledgeGraphCausalReasoner(CausalReasoner):
    """Check causal relationships using a knowledge graph."""

    def __init__(self, graph: dict[str, Iterable[str]]):
        self.associator = MultiHopAssociator(graph)

    def check_causality(self, cause: str, effect: str) -> Tuple[bool, Iterable[str]]:
        path = self.associator.find_path(cause, effect)
        return bool(path), path


class CounterfactualGraphReasoner(CounterfactualReasoner):
    """Simple counterfactual reasoning based on causal paths."""

    def __init__(self, causal_reasoner: KnowledgeGraphCausalReasoner):
        self.causal_reasoner = causal_reasoner

    def evaluate_counterfactual(self, cause: str, effect: str) -> str:
        exists, path = self.causal_reasoner.check_causality(cause, effect)
        if not exists:
            return f"{cause} has no causal effect on {effect}."
        chain = " -> ".join(path)
        return f"Without {cause}, {effect} would not occur via {chain}."

