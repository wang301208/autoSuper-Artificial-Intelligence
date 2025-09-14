"""High level wrapper around the :mod:`modules.brain.limbic` module."""

from __future__ import annotations

from schemas.emotion import EmotionalState
from modules.brain.limbic import LimbicSystem


class EmotionEngine:
    """Facade exposing a simple ``process_emotion`` API."""

    def __init__(self, limbic_system: LimbicSystem | None = None) -> None:
        self.limbic_system = limbic_system or LimbicSystem()

    def process_emotion(self, stimulus: str) -> EmotionalState:
        """Return an :class:`EmotionalState` derived from ``stimulus``."""

        return self.limbic_system.react(stimulus)


__all__ = ["EmotionEngine"]

