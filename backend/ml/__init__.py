"""AutoGPT machine learning utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .backends import get_backend
from .hf_models import load_bert, load_gpt, load_vit


@dataclass
class TrainingConfig:
    """Configuration for incremental training.

    This configuration now supports multiple optimizers, optional learning
    rate scheduling and an early stopping mechanism. Hooks for adversarial and
    curriculum learning strategies can be toggled via this configuration as
    well.
    """

    initial_lr: float = 1e-4
    optimizer: str = "adam"
    lr_scheduler: str | None = None
    early_stopping_patience: int | None = None
    use_adversarial: bool = False
    use_curriculum: bool = False
    train_after_samples: int = 100
    checkpoint_dir: Path = Path("data") / "checkpoints"


DEFAULT_TRAINING_CONFIG = TrainingConfig()

__all__ = [
    "get_backend",
    "TrainingConfig",
    "DEFAULT_TRAINING_CONFIG",
    "load_gpt",
    "load_bert",
    "load_vit",
]
