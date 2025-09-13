"""Generic Genetic Algorithm implementation.

This module provides a simple genetic algorithm supporting real-valued
representations with tournament selection, one-point crossover, and Gaussian
mutation.  Individuals are repaired to respect variable bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple
import random

# Optional plugin support for multi-metric fitness evaluation
from .fitness_plugins import load_from_config


def _clip(value: float, low: float, high: float) -> float:
    """Clip *value* to the inclusive range [low, high]."""
    return max(low, min(high, value))


@dataclass
class GAConfig:
    population_size: int = 50
    tournament_size: int = 3
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1
    mutation_sigma: float = 0.1


class GeneticAlgorithm:
    """A simple real-valued genetic algorithm."""

    def __init__(
        self,
        fitness_fn: Callable[[Sequence[float]], float] | None = None,
        bounds: Sequence[Tuple[float, float]] | None = None,
        config: GAConfig | None = None,
        metric_config: str | None = None,
        metrics: Sequence[Tuple[Callable[[Sequence[float]], float], float]] | None = None,
    ) -> None:
        # ``fitness_fn`` is used when a single evaluation function is supplied.
        # ``metrics`` / ``metric_config`` enable weighted multi-metric fitness.
        if metrics is None and metric_config is not None:
            metrics = load_from_config(metric_config)

        self.metrics = list(metrics) if metrics is not None else None
        if self.metrics is None:
            if fitness_fn is None:
                raise ValueError("Either fitness_fn or metrics must be provided")
            assert bounds is not None
            self.fitness_fn = fitness_fn
        else:
            assert bounds is not None
            # Compose a fitness function from the metrics and weights.
            self.fitness_fn = lambda ind: sum(
                weight * fn(ind) for fn, weight in self.metrics  # type: ignore[misc]
            )
        self.bounds = list(bounds) if bounds is not None else []
        self.config = config or GAConfig()
        self.num_genes = len(self.bounds)
        self.best_individual: List[float] | None = None
        self.best_fitness: float | None = None

    # Population utilities -------------------------------------------------
    def _random_individual(self) -> List[float]:
        return [
            random.uniform(low, high) for (low, high) in self.bounds
        ]

    def _repair(self, individual: List[float]) -> None:
        for i, (low, high) in enumerate(self.bounds):
            individual[i] = _clip(individual[i], low, high)

    def _evaluate(self, population: List[List[float]]) -> List[float]:
        return [self.fitness_fn(ind) for ind in population]

    # Genetic operators ----------------------------------------------------
    def _tournament_selection(
        self, population: List[List[float]], fitnesses: List[float]
    ) -> List[float]:
        size = self.config.tournament_size
        participants = random.sample(range(len(population)), size)
        best_idx = max(participants, key=lambda idx: fitnesses[idx])
        return population[best_idx][:]

    def _crossover(
        self, parent1: List[float], parent2: List[float]
    ) -> Tuple[List[float], List[float]]:
        if random.random() >= self.config.crossover_rate or self.num_genes == 1:
            return parent1[:], parent2[:]
        point = random.randrange(1, self.num_genes)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def _mutate(self, individual: List[float]) -> None:
        for i in range(self.num_genes):
            if random.random() < self.config.mutation_rate:
                sigma = self.config.mutation_sigma
                individual[i] += random.gauss(0.0, sigma)
        self._repair(individual)

    # Public API -----------------------------------------------------------
    def run(self, generations: int) -> Tuple[List[float], float]:
        population = [
            self._random_individual() for _ in range(self.config.population_size)
        ]
        fitnesses = self._evaluate(population)
        self._update_best(population, fitnesses)

        for _ in range(generations):
            new_population: List[List[float]] = []
            while len(new_population) < self.config.population_size:
                p1 = self._tournament_selection(population, fitnesses)
                p2 = self._tournament_selection(population, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                self._mutate(c1)
                self._mutate(c2)
                new_population.extend([c1, c2])
            population = new_population[: self.config.population_size]
            fitnesses = self._evaluate(population)
            self._update_best(population, fitnesses)

        assert self.best_individual is not None
        assert self.best_fitness is not None
        return self.best_individual, self.best_fitness

    def _update_best(
        self, population: List[List[float]], fitnesses: List[float]
    ) -> None:
        best_idx = max(range(len(population)), key=lambda i: fitnesses[i])
        best_fit = fitnesses[best_idx]
        if self.best_fitness is None or best_fit > self.best_fitness:
            self.best_fitness = best_fit
            self.best_individual = population[best_idx][:]
