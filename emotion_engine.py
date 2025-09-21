"""High level wrapper around the :mod:`modules.brain.limbic` module."""

from __future__ import annotations

from typing import Dict, Optional

from schemas.emotion import EmotionalState
from modules.brain.limbic import LimbicSystem
from modules.brain.state import BrainRuntimeConfig


class EmotionEngine:
    """Facade exposing a simple ``process_emotion`` API."""

    def __init__(self, limbic_system: LimbicSystem | None = None, config: BrainRuntimeConfig | None = None) -> None:
        self.config = config or BrainRuntimeConfig()
        self.limbic_system = limbic_system or LimbicSystem()

    def process_emotion(self, stimulus: str, context: Optional[Dict[str, float]] = None) -> EmotionalState:
        """Return an :class:`EmotionalState` derived from ``stimulus``."""

        return self.limbic_system.react(stimulus, context=context, config=self.config)

    def update_config(self, config: BrainRuntimeConfig) -> None:
        """Update runtime configuration for subsequent emotion evaluations."""

        self.config = config


__all__ = ["EmotionEngine"]

