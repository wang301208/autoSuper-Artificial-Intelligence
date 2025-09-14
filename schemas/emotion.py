"""Basic dataclasses and enums describing emotional state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EmotionType(Enum):
    """Enumeration of simple emotion categories."""

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"


@dataclass
class EmotionalState:
    """Represents the current emotional appraisal of a stimulus."""

    emotion: EmotionType
    intensity: float = 0.0


__all__ = ["EmotionType", "EmotionalState"]

