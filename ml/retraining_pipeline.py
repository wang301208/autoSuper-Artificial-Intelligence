"""Automated model retraining pipeline.

This module accumulates new log data, retrains the model, evaluates it
against the current baseline, and deploys the new model if it performs
better.  It is intended to be triggered periodically (for example by a
cron job).
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path
import subprocess
import warnings
from typing import Dict

import pandas as pd

DATA_DIR = Path("data")
DATASET = DATA_DIR / "dataset.csv"
NEW_LOGS = DATA_DIR / "new_logs.csv"
ARTIFACTS = Path("artifacts")
CURRENT = ARTIFACTS / "current"


def accumulate_logs() -> None:
    """Append new logs to the main dataset and clear the buffer file.

    New logs are expected in ``data/new_logs.csv`` with columns
    ``state, ability, input, output, reward``. They are appended to
    ``data/dataset.csv`` which serves as the aggregated dataset for training.
    The buffer file is removed afterwards.
    """
    if not NEW_LOGS.exists():
        return

    DATA_DIR.mkdir(exist_ok=True)
    columns = ["state", "ability", "input", "output", "reward"]
    if DATASET.exists():
        df = pd.read_csv(DATASET)
    else:
        df = pd.DataFrame(columns=columns)
    new_df = pd.read_csv(NEW_LOGS)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(DATASET, index=False)
    NEW_LOGS.unlink()


def parse_metrics(metrics_file: Path) -> Dict[str, float]:
    """Parse a metrics file into a dictionary of metric values."""
    metrics: Dict[str, float] = {}
    with open(metrics_file, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            try:
                metrics[key.strip().lower().replace(" ", "_")] = float(
                    value.strip()
                )
            except ValueError:
                continue
    return metrics


def train_and_evaluate(version: str) -> Path:
    """Train a model using the aggregated dataset and return metrics path."""
    subprocess.run(
        [
            sys.executable,
            "ml/train_models.py",
            str(DATASET),
            "--model",
            "linear",
            "--version",
            version,
        ],
        check=True,
    )
    return ARTIFACTS / version / "metrics.txt"


def deploy_if_better(version: str, metrics_file: Path) -> None:
    """Deploy the trained model if it outperforms the current baseline."""

    thresholds = {
        "success_rate": 0.01,  # require at least 1% improvement
        "reward_mean": 0.0,  # reward must not decrease
        "test_mse": 0.0,  # error must be lower
    }

    new_metrics = parse_metrics(metrics_file)
    baseline_file = CURRENT / "metrics.txt"
    baseline_metrics = (
        parse_metrics(baseline_file) if baseline_file.exists() else {}
    )

    regressions = []
    for metric, threshold in thresholds.items():
        if metric not in new_metrics or metric not in baseline_metrics:
            continue
        new_val = new_metrics[metric]
        base_val = baseline_metrics[metric]
        if metric == "test_mse":
            if new_val > base_val * (1 - threshold):
                regressions.append(
                    f"{metric} {new_val:.4f} >= {base_val:.4f}"
                )
        else:
            if new_val < base_val * (1 + threshold):
                regressions.append(
                    f"{metric} {new_val:.4f} <= {base_val:.4f}"
                )

    if regressions:
        msg = (
            "Model not deployed due to metric regression: "
            + "; ".join(regressions)
        )
        warnings.warn(msg)
        print(msg)
        return

    if CURRENT.exists():
        shutil.rmtree(CURRENT)
    shutil.copytree(ARTIFACTS / version, CURRENT)
    print(f"Deployed new model version {version} with metrics {new_metrics}")


def main() -> None:
    accumulate_logs()
    if not DATASET.exists():
        print("No dataset available for training")
        return
    version = datetime.utcnow().strftime("v%Y%m%d%H%M%S")
    metrics_file = train_and_evaluate(version)
    deploy_if_better(version, metrics_file)


if __name__ == "__main__":
    main()
