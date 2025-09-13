"""Example script demonstrating multi-metric fitness extension."""

from __future__ import annotations

from evolution.generic_ga import GAConfig, GeneticAlgorithm
from evolution import fitness_plugins

# Register a custom metric to demonstrate extensibility
@fitness_plugins.register("custom_metric")
def custom_metric(individual):
    # Example: favor individuals where the third gene is close to 1
    return -abs(individual[2] - 1.0)


def main() -> None:
    bounds = [(-5, 5), (-5, 5), (-5, 5)]
    metrics = fitness_plugins.load_from_config("config/ga_example.yaml")
    # Include our custom metric with a small weight
    metrics.append((custom_metric, 0.2))
    ga = GeneticAlgorithm(bounds=bounds, metrics=metrics, config=GAConfig(population_size=30, mutation_sigma=0.5))
    best, fitness = ga.run(generations=20)
    print("Best individual:", best)
    print("Fitness:", fitness)


if __name__ == "__main__":
    main()
