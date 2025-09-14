"""Simplified limbic system components for emotion processing.

This module contains lightweight stand‑ins for brain regions typically
associated with emotional processing.  The goal of these classes is not to
model neuroscience accurately but to provide a small, easily testable API
that other parts of the project can interact with.

The :class:`LimbicSystem` orchestrates three sub‑modules:

``EmotionProcessor``
    Derives a primary emotion from a textual stimulus.
``MemoryConsolidator``
    Stores past stimulus/response pairs for later inspection.
``HomeostasisController``
    Keeps the emotional intensity within a bounded range.
"""

from __future__ import annotations

from typing import List, Tuple

from schemas.emotion import EmotionalState, EmotionType


class EmotionProcessor:
    """Very small heuristic based emotion classifier."""

    POSITIVE_KEYWORDS = {"good", "happy", "love", "great"}
    NEGATIVE_KEYWORDS = {"bad", "sad", "angry", "terrible"}

    def evaluate(self, stimulus: str) -> EmotionType:
        """Return an :class:`EmotionType` for ``stimulus``.

        The implementation uses simple keyword matching and defaults to
        ``NEUTRAL`` when no keywords are detected.
        """

        lowered = stimulus.lower()
        if any(word in lowered for word in self.POSITIVE_KEYWORDS):
            return EmotionType.HAPPY
        if any(word in lowered for word in self.NEGATIVE_KEYWORDS):
            return EmotionType.SAD
        return EmotionType.NEUTRAL


class MemoryConsolidator:
    """Store stimulus and emotion pairs for rudimentary memory."""

    def __init__(self) -> None:
        self.memory: List[Tuple[str, EmotionType]] = []

    def consolidate(self, stimulus: str, emotion: EmotionType) -> None:
        self.memory.append((stimulus, emotion))


class HomeostasisController:
    """Regulate the intensity of an :class:`EmotionalState`."""

    def regulate(self, state: EmotionalState) -> EmotionalState:
        # Clamp intensity into the inclusive range [0.0, 1.0]
        state.intensity = max(0.0, min(1.0, state.intensity))
        return state


class LimbicSystem:
    """High level facade coordinating limbic sub‑modules."""

    def __init__(self) -> None:
        self.emotion_processor = EmotionProcessor()
        self.memory_consolidator = MemoryConsolidator()
        self.homeostasis_controller = HomeostasisController()

    def react(self, stimulus: str) -> EmotionalState:
        """Process ``stimulus`` and return the resulting emotional state."""

        emotion = self.emotion_processor.evaluate(stimulus)
        state = EmotionalState(emotion=emotion, intensity=1.0)
        state = self.homeostasis_controller.regulate(state)
        self.memory_consolidator.consolidate(stimulus, emotion)
        return state


__all__ = [
    "EmotionProcessor",
    "MemoryConsolidator",
    "HomeostasisController",
    "LimbicSystem",
]

