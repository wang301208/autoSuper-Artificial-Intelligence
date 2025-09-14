from __future__ import annotations

"""Unified evolution engine coordinating cognitive evolution components.

This module defines :class:`EvolutionEngine` which ties together
:class:`SelfEvolvingCognition` and :class:`EvolvingCognitiveArchitecture`.
It exposes :meth:`run_evolution_cycle` to drive the evolution process using
performance metrics and keeps a history of all evolved architectures to enable
rollback.
"""

from typing import Dict, Iterable, List

from modules.monitoring.collector import MetricEvent

from .evolving_cognitive_architecture import EvolvingCognitiveArchitecture, GeneticAlgorithm
from .self_evolving_cognition import EvolutionRecord, SelfEvolvingCognition


class EvolutionEngine:
    """Coordinate architecture evolution based on performance metrics."""

    def __init__(
        self,
        initial_architecture: Dict[str, float],
        fitness_fn,
        ga: GeneticAlgorithm | None = None,
    ) -> None:
        self.evolver = EvolvingCognitiveArchitecture(fitness_fn, ga)
        self.cognition = SelfEvolvingCognition(initial_architecture, self.evolver)

    # ------------------------------------------------------------------
    def run_evolution_cycle(self, metrics: Iterable[MetricEvent]) -> Dict[str, float]:
        """Evolve the architecture using the provided ``metrics``.

        The metrics are aggregated into a single performance score. Candidate
        architectures are generated via the underlying genetic algorithm and
        the best one replaces the current architecture. The new architecture is
        appended to the evolution history and returned.
        """

        metrics = list(metrics)
        if not metrics:
            return self.cognition.architecture

        # Aggregate metrics into a performance score.
        performance = sum(self.cognition._score_event(m) for m in metrics) / len(metrics)

        # Generate and apply the best candidate architecture.
        new_arch = self.evolver.evolve_architecture(self.cognition.architecture, performance)
        new_perf = self.evolver.fitness_fn(new_arch)

        self.cognition.version += 1
        self.cognition.architecture = new_arch
        self.cognition.history.append(
            EvolutionRecord(self.cognition.version, new_arch.copy(), new_perf)
        )
        return new_arch

    # ------------------------------------------------------------------
    def rollback(self, version: int) -> Dict[str, float]:
        """Rollback to a previous architecture version."""

        return self.cognition.rollback(version)

    # ------------------------------------------------------------------
    def history(self) -> List[EvolutionRecord]:
        """Return the evolution history."""

        return self.cognition.history
