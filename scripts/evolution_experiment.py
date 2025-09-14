"""Experiment comparing baseline GA with exploration-enhanced version.

The script runs multiple trials of a simple optimisation problem containing
both a local and a global optimum.  It records the success rate of escaping
the local optimum as well as average best fitness across runs.
Results are written to ``modules/evolution/experiment_results.csv``.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from modules.evolution.generic_ga import GAConfig, GeneticAlgorithm

RESULTS_FILE = Path("modules/evolution/experiment_results.csv")


def objective(individual):
    x = individual[0]
    # Two Gaussian peaks: global optimum at x=1, local at x=-1
    return math.exp(-(x - 1) ** 2) + 0.8 * math.exp(-(x + 1) ** 2)


def run_experiment(runs: int = 20) -> None:
    baseline_success = 0
    baseline_fitness: list[float] = []
    enhanced_success = 0
    enhanced_fitness: list[float] = []

    for _ in range(runs):
        base_ga = GeneticAlgorithm(
            fitness_fn=objective,
            bounds=[(-2, 2)],
            config=GAConfig(population_size=30, mutation_sigma=0.3, perturbation_rate=0.0, diversity_threshold=0.0),
        )
        b_ind, b_fit = base_ga.run(generations=40)
        baseline_fitness.append(b_fit)
        if b_ind[0] > 0:
            baseline_success += 1

        exp_ga = GeneticAlgorithm(
            fitness_fn=objective,
            bounds=[(-2, 2)],
            config=GAConfig(population_size=30, mutation_sigma=0.3, perturbation_rate=0.1, diversity_threshold=0.1),
        )
        e_ind, e_fit = exp_ga.run(generations=40)
        enhanced_fitness.append(e_fit)
        if e_ind[0] > 0:
            enhanced_success += 1

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "success_rate", "avg_fitness"])
        writer.writerow(["baseline", baseline_success / runs, mean(baseline_fitness)])
        writer.writerow(["exploration", enhanced_success / runs, mean(enhanced_fitness)])
    print(f"Results written to {RESULTS_FILE}")


if __name__ == "__main__":
    run_experiment()
