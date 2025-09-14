"""Tests for SelfEvolvingAIArchitecture."""

import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.getcwd()))

from modules.evolution import (
    EvolvingCognitiveArchitecture,
    EvolutionGeneticAlgorithm,
    SelfEvolvingAIArchitecture,
    SelfEvolvingCognition,
)
from modules.evolution.evolving_cognitive_architecture import GAConfig
from modules.monitoring.collector import MetricEvent, RealTimeMetricsCollector


def fitness_fn(arch):
    x = arch["weight"]
    return -(x - 1.0) ** 2


def _basic_setup():
    random.seed(0)
    ga = EvolutionGeneticAlgorithm(
        fitness_fn, GAConfig(population_size=10, generations=5, mutation_sigma=0.5)
    )
    evolver = EvolvingCognitiveArchitecture(fitness_fn, ga)
    collector = RealTimeMetricsCollector()
    return evolver, collector


# ---------------------------------------------------------------

def test_bottleneck_analysis_to_update():
    evolver, collector = _basic_setup()
    arch = SelfEvolvingAIArchitecture({"weight": 0.0}, evolver, collector)

    collector._events.extend(
        [
            MetricEvent("A", latency=2.0, energy=0.0, throughput=1.0, timestamp=0.0),
            MetricEvent("A", latency=3.0, energy=0.0, throughput=1.0, timestamp=1.0),
            MetricEvent("B", latency=1.0, energy=0.0, throughput=1.0, timestamp=2.0),
        ]
    )

    bottlenecks = arch.analyze_performance_bottlenecks()
    assert bottlenecks[0][0] == "A"

    candidates = arch.generate_architecture_mutations()
    best = arch.evolutionary_selection(candidates)
    assert arch.architecture == best
    assert len(arch.history) > 1
    arch.rollback(0)
    assert arch.architecture == arch.history[0].architecture


# ---------------------------------------------------------------

def test_history_shared_with_cognition():
    evolver, collector = _basic_setup()
    cognition = SelfEvolvingCognition({"weight": 0.0}, evolver, collector)
    arch = SelfEvolvingAIArchitecture(
        cognition.architecture, evolver, collector, cognition
    )

    assert arch.history is cognition.history

    collector._events.append(
        MetricEvent("mod", latency=0.1, energy=0.0, throughput=1.0, timestamp=0.0)
    )
    bottlenecks = arch.analyze_performance_bottlenecks()
    candidates = arch.generate_architecture_mutations()
    arch.evolutionary_selection(candidates)
    assert cognition.architecture == arch.architecture
    arch.rollback(0)
    assert arch.architecture == cognition.architecture == arch.history[0].architecture
