import os
import sys
import csv
import json
from pathlib import Path

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.metrics.recorder import MetricsRecorder, _flatten_dict


def _create_recorder():
    recorder = MetricsRecorder()
    recorder.record(
        algorithm="alg",
        problem="prob",
        seed=1,
        best_val=1.0,
        optimum_val=1.0,
        iterations=2,
        elapsed_time=0.1,
        extra={"nested": {"a": 1}, "list": [1, 2]},
    )
    return recorder


def test_json_round_trip(tmp_path: Path) -> None:
    recorder = _create_recorder()
    path = tmp_path / "m.json"
    recorder.save(str(path))
    with open(path) as f:
        data = json.load(f)
    assert data == recorder.to_list()


def test_csv_flatten(tmp_path: Path) -> None:
    recorder = _create_recorder()
    path = tmp_path / "m.csv"
    recorder.save(str(path))
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    expected = _flatten_dict(recorder.to_list()[0])
    expected = {k: str(v) for k, v in expected.items()}
    assert rows[0] == expected


def test_yaml_round_trip(tmp_path: Path) -> None:
    try:
        import yaml  # type: ignore
    except Exception:
        pytest.skip("pyyaml not installed")
    recorder = _create_recorder()
    path = tmp_path / "m.yaml"
    recorder.save(str(path))
    with open(path) as f:
        data = yaml.safe_load(f)
    assert data == recorder.to_list()
