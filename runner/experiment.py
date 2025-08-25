"""Utility to run algorithms on benchmark problems with multiple seeds."""
from __future__ import annotations
from typing import Iterable, Dict, Any, List

import yaml

from algorithms import ALGORITHMS
from benchmarks import PROBLEMS


def _load_config(path: str = "config/experiment.yaml") -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def run_experiments(
    algorithm: str,
    problem: str,
    seeds: Iterable[int],
    max_iters: int | None = None,
    max_time: float | None = None,
    patience: int | None = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Run ``algorithm`` on ``problem`` for each seed."""
    config = _load_config()
    max_iters = max_iters or config.get("max_iters", 100)
    max_time = max_time or config.get("time_budget")
    patience = patience or config.get("patience")

    algo_fn = ALGORITHMS[algorithm]
    problem_cls = PROBLEMS[problem]
    prob = problem_cls()

    results: List[Dict[str, Any]] = []
    for seed in seeds:
        best_x, best_val = algo_fn(
            prob,
            seed=seed,
            max_iters=max_iters,
            max_time=max_time,
            patience=patience,
            **kwargs,
        )
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
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--max-time", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    output = run_experiments(
        args.algorithm,
        args.problem,
        args.seeds,
        max_iters=args.max_iters,
        max_time=args.max_time,
        patience=args.patience,
    )
    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
    else:
        print(json.dumps(output, indent=2))
