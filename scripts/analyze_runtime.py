"""Analyze instrumentation logs to derive parameter suggestions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def load_metrics(path: Path) -> list[dict]:
    metrics: list[dict] = []
    if not path.exists():
        return metrics
    for line in path.read_text().splitlines():
        if "runtime_metrics" not in line:
            continue
        data = line.split("runtime_metrics", 1)[1].strip()
        try:
            metrics.append(json.loads(data))
        except json.JSONDecodeError:
            continue
    return metrics


def suggest_parameters(metrics: list[dict]) -> dict:
    if not metrics:
        return {}
    durations = [m.get("duration", 0) for m in metrics]
    avg = mean(durations)
    return {
        "average_duration": avg,
        "suggested_parameters": {"execution_timeout": round(avg * 1.5, 2)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze runtime log")
    parser.add_argument("--log", default="autogpt.log", help="Path to log file")
    args = parser.parse_args()
    metrics = load_metrics(Path(args.log))
    results = suggest_parameters(metrics)
    Path("analytics_output.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
