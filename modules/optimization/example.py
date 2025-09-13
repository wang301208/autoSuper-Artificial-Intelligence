"""Example usage of the optimization module."""
from __future__ import annotations

from modules.optimization import log_run, optimize_params


def dummy_algorithm(x: int, y: int) -> float:
    """A placeholder objective function returning a score."""
    return 1.0 / (abs(x - 3) + 1) + y / 100


def main() -> None:
    search_space = {"x": [1, 2, 3, 4], "y": [10, 20, 30]}
    params = optimize_params("dummy", search_space)
    score = dummy_algorithm(**params)
    log_run("dummy", params, {"score": score})
    print("run with", params, "-> score", score)


if __name__ == "__main__":
    main()
