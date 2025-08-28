import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.getcwd()))

from evolution.policy_trainer import PolicyTrainer


def test_policy_trainer_updates_from_rewards(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("state,action,reward\nS,A,1\nS,B,-1\n")

    policy_path = tmp_path / "policy.json"
    trainer = PolicyTrainer(dataset_path=dataset, learning_rate=0.1)
    updated = trainer.update_policy()

    assert "S" in updated
    assert updated["S"]["A"] > updated["S"]["B"]

    policy_path.write_text(json.dumps(updated))
    loaded = json.loads(policy_path.read_text())
    assert loaded["S"]["A"] == updated["S"]["A"]
