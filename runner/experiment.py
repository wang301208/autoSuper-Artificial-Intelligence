"""Utility to run algorithms on benchmark problems with multiple seeds."""
from __future__ import annotations

from typing import Iterable, Dict, Any, List

from algorithms import ALGORITHMS
from benchmarks import PROBLEMS


def run_experiments(
    algorithm: str,
    problem: str,
    seeds: Iterable[int],
    max_iters: int = 100,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Run ``algorithm`` on ``problem`` for each seed."""
    algo_fn = ALGORITHMS[algorithm]
    problem_cls = PROBLEMS[problem]
    prob = problem_cls()

    results: List[Dict[str, Any]] = []
    for seed in seeds:
        best_x, best_val = algo_fn(prob, seed=seed, max_iters=max_iters, **kwargs)
        results.append(
            {"seed": seed, "best_x": best_x.tolist(), "best_val": float(best_val)}
        )
    return results


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run optimization experiments")
    parser.add_argument("algorithm", choices=ALGORITHMS.keys())
    parser.add_argument("problem", choices=PROBLEMS.keys())
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-iters", type=int, default=100)
    args = parser.parse_args()

    output = run_experiments(
        args.algorithm, args.problem, args.seeds, max_iters=args.max_iters
    )
    print(json.dumps(output, indent=2))
