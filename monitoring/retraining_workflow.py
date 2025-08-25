"""Automated data aggregation, model training and deployment workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

from .dataset_buffer import DatasetBuffer

BASELINE_PATH = Path("models/baseline.json")


def aggregate_data(buffer: DatasetBuffer) -> list[Dict]:
    """Collect all logs from *buffer* for training."""
    return buffer.fetch_logs()


def train_model(logs: Iterable[Dict]) -> Dict[str, float]:
    """Train a model from *logs* and return its metrics.

    This placeholder implementation calculates the success rate where each
    log contains a boolean ``success`` field.
    """
    logs_list = list(logs)
    total = len(logs_list)
    successes = sum(1 for log in logs_list if log.get("success"))
    metric = successes / total if total else 0.0
    return {"metric": metric}


def evaluate_and_deploy(model: Dict[str, float], baseline_path: Path = BASELINE_PATH) -> bool:
    """Deploy *model* if its metric exceeds the baseline metric.

    Returns ``True`` if the model was deployed.
    """
    baseline_metric = 0.0
    if baseline_path.exists():
        baseline_metric = json.loads(baseline_path.read_text()).get("metric", 0.0)
    if model["metric"] > baseline_metric:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(model))
        return True
    return False


def main() -> None:
    buffer = DatasetBuffer()
    logs = aggregate_data(buffer)
    model = train_model(logs)
    improved = evaluate_and_deploy(model)
    if improved:
        buffer.clear()


if __name__ == "__main__":
    main()
