"""A very small Ant Colony Optimization style optimizer."""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from benchmarks.problems import Problem


def optimize(
    problem: Problem,
    seed: Optional[int] = None,
    max_iters: int = 100,
    ant_count: int = 20,
    q: float = 0.1,
) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)

    mean = np.array([(b[0] + b[1]) / 2 for b in problem.bounds])
    std = np.array([(b[1] - b[0]) / 2 for b in problem.bounds])

    best = None
    best_val = float("inf")

    for _ in range(max_iters):
        ants = rng.normal(mean, std, size=(ant_count, problem.dim))
        ants = problem.clip(ants)
        values = np.array([problem.evaluate(a) for a in ants])
        idx = np.argmin(values)
        if values[idx] < best_val:
            best_val = values[idx]
            best = ants[idx]
        # update pheromone (mean) towards best ant
        mean = (1 - q) * mean + q * best
        std *= 0.95  # slowly reduce exploration

    return best, best_val
