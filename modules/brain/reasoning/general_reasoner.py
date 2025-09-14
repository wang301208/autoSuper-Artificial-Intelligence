"""General reasoning engine combining several simplistic techniques.

This module defines :class:`GeneralReasoner`, a light‑weight component that
combines three complementary approaches:

* a tiny **concept graph** storing explicit relations between concepts,
* the existing :class:`AnalogicalReasoner` for structural mapping, and
* a very small **few‑shot memory** used as nearest‑neighbour retrieval.

The goal is not to provide state of the art reasoning but to offer a minimal
and fully deterministic module that can generate hypotheses even when no prior
knowledge exists.  The public :func:`reason_about_unknown` method accepts a
free‑form description, proposes hypotheses based on the available knowledge and
returns a list of steps containing both the hypothesis and the verification
attempt.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .analogical import AnalogicalReasoner


class GeneralReasoner:
    """Toy general reasoner combining graph lookup, analogy and few‑shot hints."""

    def __init__(self) -> None:
        # concept_graph maps a concept to a set of related concepts
        self.concept_graph: Dict[str, set[str]] = {}
        self.analogical = AnalogicalReasoner()
        # store few‑shot examples as ``(description, solution)`` tuples
        self.examples: List[Tuple[str, str]] = []

    # ------------------------------------------------------------------
    # knowledge insertion helpers
    def add_concept_relation(self, concept: str, related: str) -> None:
        """Insert a directed relation ``concept -> related`` into the graph."""

        self.concept_graph.setdefault(concept.lower(), set()).add(related.lower())

    def add_example(self, description: str, solution: str) -> None:
        """Store a textual example and its solution for few‑shot retrieval."""

        self.examples.append((description, solution))

    # ------------------------------------------------------------------
    def _nearest_example(self, task_description: str) -> Tuple[str, str] | None:
        """Return the example with maximal word overlap to ``task_description``."""

        if not self.examples:
            return None
        td_words = set(task_description.lower().split())
        best: Tuple[str, str] | None = None
        best_score = -1
        for desc, sol in self.examples:
            score = len(td_words & set(desc.lower().split()))
            if score > best_score:
                best = (desc, sol)
                best_score = score
        return best

    # ------------------------------------------------------------------
    def reason_about_unknown(self, task_description: str, max_steps: int = 3) -> List[Dict[str, str]]:
        """Generate hypotheses for an unfamiliar ``task_description``.

        The method tries three techniques in order: concept graph lookup,
        analogical transfer and few‑shot retrieval.  Each produced step is a
        dictionary with ``hypothesis`` and ``verification`` fields describing
        the reasoning step and how it was (naively) validated.  At least one
        step is always returned so that callers can display some form of
        progress even without prior knowledge.
        """

        steps: List[Dict[str, str]] = []
        tokens = task_description.lower().split()

        # 1) concept graph lookup – only one step is generated to keep space
        for tok in tokens:
            neighbours = self.concept_graph.get(tok)
            if neighbours:
                target = next(iter(neighbours))
                steps.append(
                    {
                        "hypothesis": f"{tok} relates to {target}",
                        "verification": "relation derived from concept graph",
                    }
                )
                break  # limit to a single graph based step
        if len(steps) >= max_steps:
            return steps[:max_steps]

        # 2) analogical reasoning if knowledge exists
        target_structure = {"subject": tokens[0] if tokens else "", "object": tokens[-1] if tokens else ""}
        mapping = self.analogical.transfer_knowledge("default", task_description, target_structure)
        if mapping and len(steps) < max_steps:
            steps.append(
                {
                    "hypothesis": f"analogy suggests mapping {mapping}",
                    "verification": "mapped roles via analogical reasoning",
                }
            )
        if len(steps) >= max_steps:
            return steps[:max_steps]

        # 3) few‑shot retrieval for solution hints
        ex = self._nearest_example(task_description)
        if ex and len(steps) < max_steps:
            desc, sol = ex
            steps.append(
                {
                    "hypothesis": f"similar to example '{desc}'",
                    "verification": f"candidate solution: {sol}",
                }
            )
        if steps:
            return steps[:max_steps]

        # 4) fallback when no knowledge exists at all
        return [
            {
                "hypothesis": "no prior knowledge available",
                "verification": "proceed with exploratory experiments",
            }
        ]


__all__ = ["GeneralReasoner"]
