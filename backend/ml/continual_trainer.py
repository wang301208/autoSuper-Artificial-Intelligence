from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

from . import DEFAULT_TRAINING_CONFIG, TrainingConfig


DEFAULT_LOG_FILE = Path("data") / "new_logs.csv"


class ContinualTrainer:
    """Incrementally fine-tune models on collected experience.

    The trainer keeps track of how many samples have been processed and triggers
    training whenever a configured threshold of new samples has been reached.
    Checkpoints are written after each training run.
    """

    def __init__(self, config: TrainingConfig = DEFAULT_TRAINING_CONFIG, log_file: Path = DEFAULT_LOG_FILE) -> None:
        self.config = config
        self.log_file = log_file
        self.trained_rows = 0
        if self.log_file.exists():
            # Account for header row
            with self.log_file.open("r", newline="") as f:
                self.trained_rows = sum(1 for _ in f) - 1
        self.pending_samples = 0
        self.step = 0

    def add_sample(self, sample: Dict[str, Any]) -> None:
        """Register a new sample and trigger training if needed."""
        self.pending_samples += 1
        if self.pending_samples >= self.config.train_after_samples:
            self.train()

    def train(self) -> None:
        """Fine-tune the model on newly collected samples."""
        new_data: List[Dict[str, Any]] = []
        if self.log_file.exists():
            with self.log_file.open("r", newline="") as f:
                reader = list(csv.DictReader(f))
            new_data = reader[self.trained_rows :]
        if not new_data:
            return

        # Placeholder for model fine-tuning logic
        print(
            f"Fine-tuning on {len(new_data)} samples with lr {self.config.initial_lr}"
        )

        self.trained_rows += len(new_data)
        self.pending_samples = 0
        self.step += 1
        self._save_checkpoint()

    def _save_checkpoint(self) -> None:
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.config.checkpoint_dir / f"checkpoint_{self.step}.pt"
        ckpt_path.write_text("checkpoint placeholder")
