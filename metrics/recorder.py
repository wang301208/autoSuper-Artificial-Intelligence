from __future__ import annotations

"""Utilities for recording optimization metrics and exporting them."""

from dataclasses import dataclass, asdict
import json
import csv
from typing import List, Optional


@dataclass
class RunMetrics:
    """Container for metrics from a single optimization run."""

    algorithm: str
    problem: str
    seed: int
    best_val: float
    relative_error: float
    iterations: int
    time: float
    iter_limit_reached: bool
    time_limit_reached: bool


class MetricsRecorder:
    """Accumulates run metrics and saves them to JSON or CSV."""

    def __init__(self) -> None:
        self.records: List[RunMetrics] = []

    def record(
        self,
        algorithm: str,
        problem: str,
        seed: int,
        best_val: float,
        optimum_val: float,
        iterations: int,
        elapsed_time: float,
        max_iters: Optional[int] = None,
        max_time: Optional[float] = None,
    ) -> None:
        """Record metrics for a single run.

        ``max_iters`` and ``max_time`` are used to determine whether the run hit
        the iteration or time budget.
        """
        if optimum_val == 0:
            relative_error = abs(best_val - optimum_val)
        else:
            relative_error = abs(best_val - optimum_val) / abs(optimum_val)

        iter_limit_reached = max_iters is not None and iterations >= max_iters
        time_limit_reached = max_time is not None and elapsed_time >= max_time

        self.records.append(
            RunMetrics(
                algorithm=algorithm,
                problem=problem,
                seed=seed,
                best_val=float(best_val),
                relative_error=float(relative_error),
                iterations=int(iterations),
                time=float(elapsed_time),
                iter_limit_reached=iter_limit_reached,
                time_limit_reached=time_limit_reached,
            )
        )

    def save(self, path: str) -> None:
        """Save all recorded metrics to ``path`` as JSON or CSV."""
        if path.lower().endswith(".json"):
            with open(path, "w") as f:
                json.dump([asdict(r) for r in self.records], f, indent=2)
        elif path.lower().endswith(".csv"):
            fieldnames = list(RunMetrics.__annotations__.keys())
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in self.records:
                    writer.writerow(asdict(r))
        else:
            raise ValueError("Unsupported file format: expected .json or .csv")

    def to_list(self) -> List[dict]:
        """Return recorded metrics as list of dictionaries."""
        return [asdict(r) for r in self.records]
