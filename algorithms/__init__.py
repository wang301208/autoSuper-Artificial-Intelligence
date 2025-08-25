"""Collection of optimization algorithms with a unified interface."""

from . import ga, pso, aco, random_search

ALGORITHMS = {
    "ga": ga.optimize,
    "pso": pso.optimize,
    "aco": aco.optimize,
    "random": random_search.optimize,
}

__all__ = ["ALGORITHMS", "ga", "pso", "aco", "random_search"]
