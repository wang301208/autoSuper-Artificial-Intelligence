from __future__ import annotations

"""Simple evaluation comparing manual and adaptive fitness functions."""

import random
from typing import Dict, Tuple

from modules.evolution.generic_ga import GAConfig, GeneticAlgorithm
from modules.evolution.fitness_adaptor import AdaptiveFitnessGenerator


# Objective functions ---------------------------------------------------------
def performance(individual: Tuple[float, float]) -> float:
    """Quadratic bowl with optimum at ``x=1``."""

    x, _ = individual
    return 1 - (x - 1) ** 2


def low_resource(individual: Tuple[float, float]) -> float:
    """Preference for small absolute ``y`` values."""

    _, y = individual
    return -abs(y)


def ethics(individual: Tuple[float, float]) -> float:
    """Penalise when ``x + y`` exceeds 1."""

    x, y = individual
    return -max(0.0, x + y - 1)


# Evaluation helpers ---------------------------------------------------------
def run_manual() -> Tuple[Tuple[float, float], float]:
    metrics = [
        (performance, 0.5),
        (low_resource, 0.3),
        (ethics, 0.2),
    ]
    ga = GeneticAlgorithm(
        metrics=metrics,
        bounds=[(-2.0, 2.0), (-2.0, 2.0)],
        config=GAConfig(population_size=30),
    )
    return ga.run(30)


def run_adaptive() -> Tuple[Tuple[float, float], float, Dict[str, float]]:
    objectives = {
        "performance": performance,
        "resource": low_resource,
        "ethics": ethics,
    }
    generator = AdaptiveFitnessGenerator(objectives)
    ga = GeneticAlgorithm(
        fitness_fn=generator.evaluate,
        bounds=[(-2.0, 2.0), (-2.0, 2.0)],
        config=GAConfig(population_size=30),
    )

    population = [ga._random_individual() for _ in range(ga.config.population_size)]
    fitnesses = ga._evaluate(population)
    ga._update_best(population, fitnesses)

    for generation in range(30):
        if generation > 15:
            # After half the run, emphasise ethics via environment signal
            generator.update_environment({"ethics": 1.0})
        new_population = []
        while len(new_population) < ga.config.population_size:
            p1 = ga._tournament_selection(population, fitnesses)
            p2 = ga._tournament_selection(population, fitnesses)
            c1, c2 = ga._crossover(p1, p2)
            ga._mutate(c1)
            ga._mutate(c2)
            new_population.extend([c1, c2])
        population = new_population[: ga.config.population_size]
        fitnesses = ga._evaluate(population)
        ga._update_best(population, fitnesses)

    assert ga.best_individual is not None
    assert ga.best_fitness is not None
    return ga.best_individual, ga.best_fitness, generator.weights


def main() -> None:
    random.seed(0)
    manual_best, manual_fit = run_manual()
    adaptive_best, adaptive_fit, weights = run_adaptive()
    print("Manual best fitness:", round(manual_fit, 4), "individual:", manual_best)
    print(
        "Adaptive best fitness:",
        round(adaptive_fit, 4),
        "individual:",
        adaptive_best,
        "weights:",
        {k: round(v, 3) for k, v in weights.items()},
    )


if __name__ == "__main__":  # pragma: no cover - convenience script
    main()
