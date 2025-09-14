"""Tests for the SelfEvolvingCognition module."""

import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.getcwd()))

from modules.evolution import (
    EvolvingCognitiveArchitecture,
    EvolutionGeneticAlgorithm,
    SelfEvolvingCognition,
)
from modules.evolution.evolving_cognitive_architecture import GAConfig
from modules.monitoring.collector import MetricEvent, RealTimeMetricsCollector


def fitness_fn(arch):
    x = arch["weight"]
    return -(x - 1.0) ** 2


def test_long_term_evolution_improves_performance():
    random.seed(0)
    ga = EvolutionGeneticAlgorithm(fitness_fn, GAConfig(population_size=10, generations=5, mutation_sigma=0.5))
    evolver = EvolvingCognitiveArchitecture(fitness_fn, ga)
    collector = RealTimeMetricsCollector()
    cognition = SelfEvolvingCognition({"weight": 0.0}, evolver, collector)

    for step in range(5):
        perf = fitness_fn(cognition.architecture)
        event = MetricEvent(
            module="evolve", latency=0.0, energy=0.0, throughput=perf, timestamp=float(step)
        )
        collector._events.append(event)
        cognition.observe()

    performances = [rec.performance for rec in cognition.history]
    assert performances[-1] > performances[0]
    cognition.rollback(0)
    assert cognition.architecture == cognition.history[0].architecture
    diff = cognition.compare(0, len(cognition.history) - 1)
    assert diff["performance_diff"] == performances[-1] - performances[0]
