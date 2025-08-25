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

import pandas as pd

DATA_DIR = Path("data")
DATASET = DATA_DIR / "dataset.csv"
NEW_LOGS = DATA_DIR / "new_logs.csv"
ARTIFACTS = Path("artifacts")
CURRENT = ARTIFACTS / "current"


def accumulate_logs() -> None:
    """Append new logs to the main dataset and clear the buffer file.

    New logs are expected in ``data/new_logs.csv`` with columns ``text`` and
    ``target``.  They are appended to ``data/dataset.csv`` which serves as the
    aggregated dataset for training.  The buffer file is removed afterwards.
    """
    if not NEW_LOGS.exists():
        return

    DATA_DIR.mkdir(exist_ok=True)
    if DATASET.exists():
        df = pd.read_csv(DATASET)
    else:
        df = pd.DataFrame(columns=["text", "target"])
    new_df = pd.read_csv(NEW_LOGS)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(DATASET, index=False)
    NEW_LOGS.unlink()


def parse_test_mse(metrics_file: Path) -> float:
    with open(metrics_file, "r") as f:
        for line in f:
            if line.startswith("Test MSE:"):
                return float(line.split(":", 1)[1].strip())
    raise ValueError("Test MSE not found in metrics file")


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
    new_metric = parse_test_mse(metrics_file)

    baseline_metric = float("inf")
    baseline_file = CURRENT / "metrics.txt"
    if baseline_file.exists():
        baseline_metric = parse_test_mse(baseline_file)

    if new_metric < baseline_metric:
        if CURRENT.exists():
            shutil.rmtree(CURRENT)
        shutil.copytree(ARTIFACTS / version, CURRENT)
        print(
            f"Deployed new model version {version} (Test MSE {new_metric:.4f})"
        )
    else:
        print(
            f"Model not deployed: {new_metric:.4f} >= {baseline_metric:.4f}"
        )


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
