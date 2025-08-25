"""Baseline random search optimizer."""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from benchmarks.problems import Problem


def optimize(
    problem: Problem,
    seed: Optional[int] = None,
    max_iters: int = 1000,
) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    lower = np.array([b[0] for b in problem.bounds])
    upper = np.array([b[1] for b in problem.bounds])
    best = None
    best_val = float("inf")
    for _ in range(max_iters):
        x = rng.uniform(lower, upper)
        val = problem.evaluate(x)
        if val < best_val:
            best_val = val
            best = x
    return best, best_val
