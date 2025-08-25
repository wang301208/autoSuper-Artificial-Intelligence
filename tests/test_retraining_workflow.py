import json
from pathlib import Path

from monitoring.dataset_buffer import DatasetBuffer
from monitoring.retraining_workflow import (
    aggregate_data,
    evaluate_and_deploy,
    train_model,
)


def test_dataset_buffer(tmp_path):
    db_path = tmp_path / "buffer.db"
    buffer = DatasetBuffer(db_path)
    buffer.add_log({"success": True})
    buffer.add_log({"success": False})
    logs = aggregate_data(buffer)
    assert len(logs) == 2
    model = train_model(logs)
    assert model["metric"] == 0.5
    buffer.close()


def test_evaluate_and_deploy(tmp_path):
    baseline = tmp_path / "baseline.json"
    # First deployment should succeed because baseline doesn't exist
    model = {"metric": 0.8}
    assert evaluate_and_deploy(model, baseline) is True
    assert json.loads(baseline.read_text())["metric"] == 0.8
    # Worse model should not be deployed
    assert evaluate_and_deploy({"metric": 0.5}, baseline) is False
    assert json.loads(baseline.read_text())["metric"] == 0.8
