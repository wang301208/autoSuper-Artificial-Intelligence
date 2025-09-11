from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .continual_trainer import ContinualTrainer
from . import DEFAULT_TRAINING_CONFIG

LOG_FILE = Path("data") / "new_logs.csv"

# Single trainer instance used to schedule periodic fine-tuning
TRAINER = ContinualTrainer(DEFAULT_TRAINING_CONFIG, LOG_FILE)


def log_interaction(task: Any, ability: str, result: Any, reward: float) -> None:
    """Record an interaction to ``data/new_logs.csv``.

    Parameters
    ----------
    task:
        The task or state associated with this interaction.
    ability:
        Name of the ability that was executed.
    result:
        Result object from the ability execution. ``input`` and ``output``
        attributes (or keys if ``result`` is a mapping) are extracted if
        available.
    reward:
        Numeric reward representing the quality of the result.
    """
    state = getattr(task, "id", getattr(task, "name", str(task)))

    if isinstance(result, dict):
        input_data = result.get("input") or result.get("prompt") or ""
        output_data = result.get("output") or result.get("response") or str(result)
    else:
        input_data = getattr(result, "input", "")
        output_data = getattr(result, "output", str(result))

    LOG_FILE.parent.mkdir(exist_ok=True)
    file_exists = LOG_FILE.exists()
    with LOG_FILE.open("a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["state", "ability", "input", "output", "reward"]
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "state": state,
                "ability": ability,
                "input": input_data,
                "output": output_data,
                "reward": reward,
            }
        )
    TRAINER.add_sample(
        {
            "state": state,
            "ability": ability,
            "input": input_data,
            "output": output_data,
            "reward": reward,
        }
    )
