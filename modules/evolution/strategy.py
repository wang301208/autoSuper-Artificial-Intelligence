"""Exploration strategies for evolutionary algorithms.

This module provides optional operators that can be used to increase
exploration in evolutionary algorithms. Currently it implements
simulated annealing for refining individuals and an innovation protection
routine that replaces overly similar individuals with random ones.
"""
from __future__ import annotations

import math
import random
from typing import Callable, List, Sequence, Tuple


def simulated_annealing(
    individual: List[float],
    fitness_fn: Callable[[Sequence[float]], float],
    bounds: Sequence[Tuple[float, float]],
    temperature: float = 1.0,
    cooling: float = 0.95,
    steps: int = 100,
) -> Tuple[List[float], float]:
    """Refine *individual* using a simple simulated annealing scheme.

    Parameters
    ----------
    individual
        Starting point to be refined.
    fitness_fn
        Function returning a fitness value for an individual.
    bounds
        Sequence of ``(low, high)`` tuples describing valid ranges.
    temperature
        Initial temperature controlling exploration magnitude.
    cooling
        Multiplicative factor applied to the temperature each step.
    steps
        Maximum number of refinement steps.
    """
    current = individual[:]
    current_fitness = fitness_fn(current)
    best = current[:]
    best_fitness = current_fitness
    temp = temperature
    for _ in range(steps):
        candidate = [
            _clip(g + random.gauss(0.0, temp), low, high)
            for g, (low, high) in zip(current, bounds)
        ]
        candidate_fitness = fitness_fn(candidate)
        delta = candidate_fitness - current_fitness
        if delta > 0 or random.random() < math.exp(delta / max(temp, 1e-9)):
            current = candidate
            current_fitness = candidate_fitness
            if current_fitness > best_fitness:
                best = current[:]
                best_fitness = current_fitness
        temp *= cooling
        if temp < 1e-6:
            break
    return best, best_fitness


def innovation_protection(
    population: List[List[float]],
    bounds: Sequence[Tuple[float, float]],
    min_distance: float = 1e-3,
) -> List[List[float]]:
    """Maintain diversity by replacing overly similar individuals.

    Individuals closer than ``min_distance`` in Euclidean space are
    replaced with random individuals drawn uniformly from ``bounds``.
    The population is modified in place and also returned for convenience.
    """

    def distance(a: Sequence[float], b: Sequence[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    unique: List[List[float]] = []
    for i, ind in enumerate(population):
        if any(distance(ind, u) < min_distance for u in unique):
            population[i] = [random.uniform(low, high) for low, high in bounds]
            unique.append(population[i])
        else:
            unique.append(ind)
    return population


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
