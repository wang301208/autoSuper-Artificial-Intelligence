import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.getcwd()))

from modules.evolution import EvolutionEngine, EvolutionGeneticAlgorithm
from modules.evolution.evolving_cognitive_architecture import GAConfig
from modules.monitoring.collector import MetricEvent


def fitness_fn(arch):
    x = arch["weight"]
    return -(x - 1.0) ** 2


def test_multi_cycle_evolution_and_rollback():
    random.seed(0)
    ga = EvolutionGeneticAlgorithm(
        fitness_fn, GAConfig(population_size=10, generations=5, mutation_sigma=0.5)
    )
    engine = EvolutionEngine({"weight": 0.0}, fitness_fn, ga)

    # Run several evolution cycles using feedback derived from current performance
    for step in range(5):
        perf = fitness_fn(engine.cognition.architecture)
        metrics = [
            MetricEvent(
                module="evolve",
                latency=0.0,
                energy=0.0,
                throughput=perf,
                timestamp=float(step),
            )
        ]
        engine.run_evolution_cycle(metrics)

    performances = [rec.performance for rec in engine.history()]
    assert performances[-1] > performances[0]

    # Verify rollback restores initial architecture
    initial_arch = engine.history()[0].architecture
    engine.rollback(0)
    assert engine.cognition.architecture == initial_arch
