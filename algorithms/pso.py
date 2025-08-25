"""Particle Swarm Optimization with unified interface."""
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from benchmarks.problems import Problem


def optimize(
    problem: Problem,
    seed: Optional[int] = None,
    max_iters: int = 100,
    swarm_size: int = 30,
    inertia: float = 0.7,
    cognitive: float = 1.4,
    social: float = 1.4,
) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)

    lower = np.array([b[0] for b in problem.bounds])
    upper = np.array([b[1] for b in problem.bounds])

    pos = rng.uniform(lower, upper, size=(swarm_size, problem.dim))
    vel = rng.normal(0, 1, size=(swarm_size, problem.dim))
    personal_best = pos.copy()
    personal_best_val = np.array([problem.evaluate(p) for p in pos])

    best_idx = np.argmin(personal_best_val)
    global_best = personal_best[best_idx].copy()
    global_best_val = personal_best_val[best_idx]

    for _ in range(max_iters):
        r1, r2 = rng.random(size=(2, swarm_size, problem.dim))
        vel = (
            inertia * vel
            + cognitive * r1 * (personal_best - pos)
            + social * r2 * (global_best - pos)
        )
        pos = problem.clip(pos + vel)
        values = np.array([problem.evaluate(p) for p in pos])

        improved = values < personal_best_val
        personal_best[improved] = pos[improved]
        personal_best_val[improved] = values[improved]

        idx = np.argmin(personal_best_val)
        if personal_best_val[idx] < global_best_val:
            global_best_val = personal_best_val[idx]
            global_best = personal_best[idx].copy()

    return global_best, global_best_val
