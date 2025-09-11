"""AutoGPT machine learning utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .backends import get_backend


@dataclass
class TrainingConfig:
    """Configuration for incremental training."""

    initial_lr: float = 1e-4
    lr_schedule: str = "linear"
    train_after_samples: int = 100
    checkpoint_dir: Path = Path("data") / "checkpoints"


DEFAULT_TRAINING_CONFIG = TrainingConfig()

__all__ = ["get_backend", "TrainingConfig", "DEFAULT_TRAINING_CONFIG"]
