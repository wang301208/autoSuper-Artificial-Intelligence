"""Long-running experiment for AutoOptimizer.

This script simulates a system over an extended period, periodically logging
resource usage and prediction outcomes. The
:class:`~monitoring.auto_optimizer.AutoOptimizer` reacts to these metrics and
records each optimization decision along with the observed effect. Inspecting
the stored events after the run provides insight into the benefits of automatic
tuning.

Run the experiment:

    python scripts/auto_optimizer_experiment.py --duration 60

The default duration is short for demonstration purposes, but it can be
increased for more exhaustive experiments.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from monitoring import AutoOptimizer, PerformanceMonitor, TimeSeriesStorage


def run(duration: int) -> None:
    storage = TimeSeriesStorage(Path("experiment.db"))
    monitor = PerformanceMonitor(
        storage,
        training_accuracy=0.9,
        degradation_threshold=0.1,
        cpu_threshold=80.0,
        memory_threshold=80.0,
    )
    optimizer = AutoOptimizer(
        monitor,
        storage,
        cpu_threshold=80.0,
        memory_threshold=80.0,
        accuracy_threshold=0.8,
        parameter_bounds={"learning_rate": (0.001, 0.1)},
    )

    start = time.time()
    while time.time() - start < duration:
        # simulate metrics
        monitor.log_resource_usage("agent", random.uniform(0, 100), random.uniform(0, 100))
        prediction = 1 if random.random() > 0.5 else 0
        outcome = 1 if random.random() > 0.3 else 0
        monitor.log_prediction(prediction, outcome)

        optimizer.step()
        time.sleep(1)

    events = storage.events("optimization")
    print(f"Recorded {len(events)} optimization events")
    for e in events:
        print(e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AutoOptimizer experiment")
    parser.add_argument("--duration", type=int, default=10, help="Duration of experiment in seconds")
    args = parser.parse_args()
    run(args.duration)


if __name__ == "__main__":
    main()

