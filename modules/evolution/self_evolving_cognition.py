"""Self-evolving cognition module integrating performance monitoring.

This module provides :class:`SelfEvolvingCognition` which observes performance
metrics and automatically triggers evolution of a cognitive architecture using
:class:`EvolvingCognitiveArchitecture`.  After each evolution step the
performance and architecture version are recorded, enabling rollback and
comparison of past versions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from modules.monitoring.collector import MetricEvent, RealTimeMetricsCollector
from .evolving_cognitive_architecture import EvolvingCognitiveArchitecture


@dataclass
class EvolutionRecord:
    """Record of a single evolution step."""

    version: int
    architecture: Dict[str, float]
    performance: float


class SelfEvolvingCognition:
    """Automatically evolve a cognitive architecture based on performance metrics."""

    def __init__(
        self,
        initial_architecture: Dict[str, float],
        evolver: EvolvingCognitiveArchitecture,
        collector: Optional[RealTimeMetricsCollector] = None,
    ) -> None:
        self.architecture = initial_architecture
        self.evolver = evolver
        self.collector = collector
        self._processed_events = 0
        self.version = 0
        initial_perf = self.evolver.fitness_fn(initial_architecture)
        self.history: List[EvolutionRecord] = [
            EvolutionRecord(self.version, initial_architecture.copy(), initial_perf)
        ]

    # ------------------------------------------------------------------
    def _score_event(self, event: MetricEvent) -> float:
        """Derive a scalar performance score from a metric event."""

        return event.throughput - event.latency - event.energy

    # ------------------------------------------------------------------
    def _process_event(self, event: MetricEvent) -> None:
        """Process a new metric event and evolve the architecture."""

        performance = self._score_event(event)
        new_arch = self.evolver.evolve_architecture(self.architecture, performance)
        new_perf = self.evolver.fitness_fn(new_arch)
        self.version += 1
        self.architecture = new_arch
        self.history.append(
            EvolutionRecord(self.version, new_arch.copy(), new_perf)
        )

    # ------------------------------------------------------------------
    def observe(self) -> None:
        """Observe new metric events and trigger evolution steps."""

        if self.collector is None:
            return
        events = self.collector.events()
        for event in events[self._processed_events :]:
            self._process_event(event)
        self._processed_events = len(events)

    # ------------------------------------------------------------------
    def rollback(self, version: int) -> Dict[str, float]:
        """Rollback to a previous architecture version."""

        for record in self.history:
            if record.version == version:
                self.architecture = record.architecture.copy()
                self.version = record.version
                return self.architecture
        raise ValueError(f"Version {version} not found in history")

    # ------------------------------------------------------------------
    def compare(self, v1: int, v2: int) -> Dict[str, float]:
        """Compare the performance between two versions."""

        rec1 = next(r for r in self.history if r.version == v1)
        rec2 = next(r for r in self.history if r.version == v2)
        return {"performance_diff": rec2.performance - rec1.performance}
