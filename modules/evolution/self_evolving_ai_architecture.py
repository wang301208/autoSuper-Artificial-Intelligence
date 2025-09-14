"""Self-evolving AI architecture orchestrator."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from modules.monitoring.collector import RealTimeMetricsCollector
from .self_evolving_cognition import EvolutionRecord, SelfEvolvingCognition
from .evolving_cognitive_architecture import EvolvingCognitiveArchitecture


class SelfEvolvingAIArchitecture:
    """Analyse metrics and evolve an AI architecture accordingly.

    This class cooperates with :class:`SelfEvolvingCognition` and
    :class:`EvolvingCognitiveArchitecture` to share evolution history and provide
    rollback capabilities.  Metrics are collected via
    :class:`RealTimeMetricsCollector`.
    """

    def __init__(
        self,
        initial_architecture: Dict[str, float],
        evolver: EvolvingCognitiveArchitecture,
        collector: Optional[RealTimeMetricsCollector] = None,
        cognition: Optional[SelfEvolvingCognition] = None,
    ) -> None:
        self.collector = collector
        self.evolver = evolver
        self.cognition = cognition

        if cognition is not None:
            # Share history and version with the cognition module
            self.history = cognition.history
            self.version = cognition.version
            self.architecture = cognition.architecture
        else:
            self.architecture = initial_architecture
            initial_perf = self.evolver.fitness_fn(initial_architecture)
            self.version = 0
            self.history: List[EvolutionRecord] = [
                EvolutionRecord(self.version, initial_architecture.copy(), initial_perf)
            ]

    # ------------------------------------------------------------------
    def analyze_performance_bottlenecks(self) -> List[Tuple[str, float]]:
        """Identify modules with highest average latency from collected metrics."""

        if self.collector is None:
            return []
        events = self.collector.events()
        if not events:
            return []

        stats: Dict[str, List[float]] = defaultdict(list)
        for event in events:
            stats[event.module].append(event.latency)
        averages = [(module, sum(vals) / len(vals)) for module, vals in stats.items()]
        averages.sort(key=lambda x: x[1], reverse=True)
        return averages

    # ------------------------------------------------------------------
    def generate_architecture_mutations(
        self, num_candidates: Optional[int] = None
    ) -> List[Tuple[Dict[str, float], float]]:
        """Generate candidate architectures using the genetic algorithm."""

        best, best_score, history = self.evolver.ga.evolve(self.architecture)
        candidates: List[Tuple[Dict[str, float], float]] = list(history)
        if (best, best_score) not in candidates:
            candidates.append((best, best_score))
        if num_candidates is not None:
            candidates = candidates[:num_candidates]
        return candidates

    # ------------------------------------------------------------------
    def evolutionary_selection(
        self, candidates: List[Tuple[Dict[str, float], float]]
    ) -> Dict[str, float]:
        """Select the best candidate based on fitness and update architecture."""

        if not candidates:
            return self.architecture
        best_arch, best_score = max(candidates, key=lambda x: x[1])
        self.update_architecture(best_arch, best_score)
        return best_arch

    # ------------------------------------------------------------------
    def update_architecture(
        self, new_arch: Dict[str, float], performance: Optional[float] = None
    ) -> None:
        """Apply ``new_arch`` and record it in the evolution history."""

        self.version += 1
        if performance is None:
            performance = self.evolver.fitness_fn(new_arch)
        self.architecture = new_arch
        record = EvolutionRecord(self.version, new_arch.copy(), performance)
        self.history.append(record)
        if self.cognition is not None:
            self.cognition.architecture = self.architecture
            self.cognition.version = self.version

    # ------------------------------------------------------------------
    def rollback(self, version: int) -> Dict[str, float]:
        """Rollback to a previous architecture version."""

        if self.cognition is not None:
            arch = self.cognition.rollback(version)
            self.architecture = arch
            self.version = self.cognition.version
            return arch
        for record in self.history:
            if record.version == version:
                self.architecture = record.architecture.copy()
                self.version = record.version
                return self.architecture
        raise ValueError(f"Version {version} not found in history")

    # ------------------------------------------------------------------
    def get_history(self) -> List[EvolutionRecord]:
        """Return the evolution history."""

        return self.history
