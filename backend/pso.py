"""Basic Particle Swarm Optimization (PSO) implementation.

This module provides a simple implementation of the Particle Swarm
Optimization algorithm suitable for educational purposes and small
optimization tasks. The algorithm follows the canonical PSO update rules
with inertia weight and supports basic strategies to prevent particle
stagnation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple

import numpy as np


@dataclass
class PSOResult:
    """Result of a PSO run.

    Attributes:
        position: The best position found.
        value: Objective function value at ``position``.
    """

    position: np.ndarray
    value: float


def pso(
    f: Callable[[np.ndarray], float],
    bounds: Iterable[Tuple[float, float]],
    num_particles: int = 30,
    max_iter: int = 100,
    w: float = 0.9,
    c1: float = 2.0,
    c2: float = 2.0,
) -> PSOResult:
    """Run Particle Swarm Optimization on ``f``.

    Args:
        f: Objective function to minimise. It must accept a NumPy array and
            return a scalar.
        bounds: Iterable of ``(lower, upper)`` tuples defining the search
            space for each dimension.
        num_particles: Number of particles in the swarm.
        max_iter: Maximum number of iterations to perform.
        w: Inertia weight controlling impact of previous velocity.
        c1: Cognitive acceleration coefficient.
        c2: Social acceleration coefficient.

    Returns:
        ``PSOResult`` containing the best position and its fitness value.
    """

    bounds = np.array(list(bounds), dtype=float)
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    dim = len(bounds)

    rng = np.random.default_rng()
    x = rng.uniform(lower, upper, size=(num_particles, dim))
    v = np.zeros_like(x)

    # Evaluate initial population
    p_best = x.copy()
    p_best_val = np.apply_along_axis(f, 1, x)
    best_idx = np.argmin(p_best_val)
    g_best = p_best[best_idx].copy()
    g_best_val = p_best_val[best_idx]

    for _ in range(max_iter):
        r1 = rng.random((num_particles, dim))
        r2 = rng.random((num_particles, dim))

        v = w * v + c1 * r1 * (p_best - x) + c2 * r2 * (g_best - x)
        x = x + v

        # Boundary handling (clamp)
        x = np.clip(x, lower, upper)

        # Evaluate and update bests
        values = np.apply_along_axis(f, 1, x)
        improved = values < p_best_val
        p_best[improved] = x[improved]
        p_best_val[improved] = values[improved]

        best_idx = np.argmin(p_best_val)
        if p_best_val[best_idx] < g_best_val:
            g_best_val = p_best_val[best_idx]
            g_best = p_best[best_idx].copy()

        # Optional inertia weight decay to encourage convergence
        w *= 0.99

        # Re-randomise stagnant particles
        stagnation = np.linalg.norm(v, axis=1) < 1e-5
        if np.any(stagnation):
            x[stagnation] = rng.uniform(lower, upper, size=(stagnation.sum(), dim))
            v[stagnation] = 0

    return PSOResult(position=g_best, value=float(g_best_val))
