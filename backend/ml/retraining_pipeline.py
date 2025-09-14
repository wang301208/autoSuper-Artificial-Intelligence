"""Automated model retraining pipeline.

This module accumulates new log data, retrains the model, evaluates it
against the current baseline, and deploys the new model if it performs
better.  It is intended to be triggered periodically (for example by a
cron job).
"""
from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
import subprocess
from typing import Dict

import pandas as pd
from events import create_event_bus, publish
from .data_pipeline import DataPipeline

from .compression import compress_model
from .model_registry import ModelRegistry
from . import distributed

DATA_DIR = Path("data")
DATASET = DATA_DIR / "dataset.csv"
NEW_LOGS = DATA_DIR / "new_logs.csv"
ARTIFACTS = Path("artifacts")
CURRENT = ARTIFACTS / "current"
PREVIOUS = ARTIFACTS / "previous"

HISTORY_FILE = Path("evolution/metrics_history.csv")
HISTORY_FIELDS = [
    "timestamp",
    "version",
    "Success Rate",
    "Mean Reward",
    "Perplexity",
    "Test MSE",
    "status",
]

event_bus = create_event_bus()

# Additional metrics considered during deployment comparison. ``direction``
# specifies whether higher values are better ("higher") or lower values are
# better ("lower"). ``threshold`` sets the minimum required improvement.
METRIC_THRESHOLDS: Dict[str, Dict[str, float | str]] = {
    "Success Rate": {"direction": "higher", "threshold": 0.0},
    "Mean Reward": {"direction": "higher", "threshold": 0.0},
}

logger = logging.getLogger(__name__)


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


def parse_metric(metrics_file: Path, key: str) -> float:
    """Extract a metric value labelled by ``key`` from ``metrics_file``."""
    with open(metrics_file, "r") as f:
        for line in f:
            if line.startswith(f"{key}:"):
                return float(line.split(":", 1)[1].strip())
    raise ValueError(f"{key} not found in metrics file")


def parse_metrics(metrics_file: Path) -> Dict[str, float]:
    """Parse a metrics file into a dictionary of metric names to values."""
    metrics: Dict[str, float] = {}
    if not metrics_file.exists():
        return metrics
    with open(metrics_file, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            try:
                metrics[key.strip()] = float(value.strip())
            except ValueError:
                continue
    return metrics


def _log_history(version: str, metrics: Dict[str, float], status: str) -> None:
    """Append metrics to the history CSV."""

    HISTORY_FILE.parent.mkdir(exist_ok=True)
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "version": version,
        "status": status,
    }
    for key in ["Success Rate", "Mean Reward", "Perplexity", "Test MSE"]:
        row[key] = metrics.get(key, float("nan"))

    write_header = not HISTORY_FILE.exists()
    with open(HISTORY_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train_and_evaluate(version: str, model: str) -> tuple[Path, str]:
    """Train a model using the aggregated dataset.

    Returns the metrics file path and the metric label to compare.
    """
    if model == "llm":
        subprocess.run(
            [
                sys.executable,
                "ml/fine_tune_llm.py",
                str(DATASET),
                "--version",
                version,
            ],
            check=True,
        )
        metric_name = "Perplexity"
    else:
        subprocess.run(
            [
                sys.executable,
                "ml/train_models.py",
                str(DATASET),
                "--model",
                model,
                "--version",
                version,
            ],
            check=True,
        )
        metric_name = "Test MSE"
    return ARTIFACTS / version / "metrics.txt", metric_name


def deploy_with_ab_test(
    version: str,
    new_metrics: Dict[str, float],
    metric_name: str,
    registry: ModelRegistry,
) -> bool:
    """Deploy the trained model and run a simple A/B test against the baseline.

    The current model is backed up, the new model is copied into ``CURRENT`` and
    compared against the previous metrics. If any regression is detected the
    deployment is rolled back and the previous model restored.
    """

    baseline_meta = registry.current()
    baseline_metrics = baseline_meta["metrics"] if baseline_meta else {}

    # Deploy new model by swapping directories so that the A/B test can run.
    if CURRENT.exists():
        if PREVIOUS.exists():
            shutil.rmtree(PREVIOUS)
        shutil.copytree(CURRENT, PREVIOUS)
        shutil.rmtree(CURRENT)
    shutil.copytree(ARTIFACTS / version, CURRENT)

    thresholds = METRIC_THRESHOLDS.copy()
    thresholds.setdefault(metric_name, {"direction": "lower", "threshold": 0.0})

    regressions: list[str] = []
    for metric, cfg in thresholds.items():
        if metric not in new_metrics or metric not in baseline_metrics:
            continue
        new_val = new_metrics[metric]
        base_val = baseline_metrics[metric]
        direction = cfg["direction"]
        threshold = float(cfg["threshold"])
        if direction == "lower":
            if new_val > base_val + threshold:
                regressions.append(f"{metric} {new_val:.4f} > {base_val:.4f}")
        else:
            if new_val < base_val - threshold:
                regressions.append(f"{metric} {new_val:.4f} < {base_val:.4f}")

    if not regressions:
        metric_val = new_metrics.get(metric_name, float("nan"))
        print(
            f"Deployed new model version {version} ({metric_name} {metric_val:.4f})",
        )
        registry.set_current(version)
        _log_history(version, new_metrics, "deployed")
        return True

    logger.warning(
        "Model not deployed due to metric regressions: %s", "; ".join(regressions)
    )
    _log_history(version, new_metrics, "regression")
    publish(
        event_bus,
        "model.regression",
        {"version": version, "regressions": regressions},
    )

    # Roll back to the previous model
    if CURRENT.exists():
        shutil.rmtree(CURRENT)
    if PREVIOUS.exists():
        shutil.copytree(PREVIOUS, CURRENT)
    registry.rollback()
    return False


def main() -> bool:
    parser = argparse.ArgumentParser(description="Retrain models on accumulated data")
    parser.add_argument(
        "--model",
        default="linear",
        help="Model type to train (e.g. 'linear' or 'llm')",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=None,
        help="Optional compression level to apply to the trained model",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed training hooks",
    )
    args = parser.parse_args()

    registry = ModelRegistry()

    accumulate_logs()
    if not DATASET.exists():
        print("No dataset available for training")
        return True

    # Expand the dataset using the data pipeline before training.
    df = pd.read_csv(DATASET)
    pipeline = DataPipeline()
    df = pipeline.process(df)
    df.to_csv(DATASET, index=False)

    version = datetime.utcnow().strftime("v%Y%m%d%H%M%S")
    if args.distributed:
        distributed.setup_training()
    metrics_file, metric_name = train_and_evaluate(version, args.model)
    if args.distributed:
        distributed.teardown_training()

    new_metrics = parse_metrics(metrics_file)
    if args.compression_level is not None:
        compress_model(ARTIFACTS / version, args.compression_level)

    registry.register(version, new_metrics, args.compression_level)
    return deploy_with_ab_test(version, new_metrics, metric_name, registry)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
