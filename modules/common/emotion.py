"""Utilities for analyzing and tracking emotional state in dialogues.

This module provides a simple :class:`EmotionAnalyzer` for classifying the
sentiment of user utterances and an :class:`EmotionState` container for
tracking the current conversational mood.  A helper function
:func:`adjust_response_style` can be used to adapt generated responses to the
latest detected emotion.

The implementation is intentionally lightweight and relies on a small
keyword‑based heuristic so that it can run in environments where heavy machine
learning dependencies are unavailable.  It can easily be replaced with a more
advanced model if desired.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


class EmotionAnalyzer:
    """Classify text into coarse emotional categories.

    The current implementation performs a very small heuristic lookup against
    sets of positive and negative keywords.  The interface is designed so that
    the classifier can later be swapped out for a real ML model without
    changing the rest of the code base.
    """

    POSITIVE_KEYWORDS = {
        "good",
        "great",
        "happy",
        "love",
        "excellent",
        "awesome",
        "fantastic",
        "amazing",
    }
    NEGATIVE_KEYWORDS = {
        "bad",
        "sad",
        "angry",
        "hate",
        "terrible",
        "awful",
        "horrible",
        "worse",
    }

    def analyze_text(self, text: str) -> str:
        """Return an emotion label for *text*.

        The label will be ``"positive"``, ``"negative"`` or ``"neutral"``
        depending on the keyword counts in the supplied text.  The comparison is
        case‑insensitive.
        """

        lowered = text.lower()
        pos = sum(word in lowered for word in self.POSITIVE_KEYWORDS)
        neg = sum(word in lowered for word in self.NEGATIVE_KEYWORDS)
        if pos > neg:
            return "positive"
        if neg > pos:
            return "negative"
        return "neutral"

    def analyze_voice(self, _audio_bytes: bytes) -> str:  # pragma: no cover - placeholder
        """Placeholder for voice analysis.

        Voice‑based sentiment analysis is outside the scope of this lightweight
        implementation.  The method exists to document the intended interface
        and may be implemented with a speech‑to‑text pipeline in the future.
        """

        raise NotImplementedError("Voice emotion analysis is not implemented")


@dataclass
class EmotionState:
    """Represents the tracked emotional state of the conversation."""

    label: str = "neutral"
    excitement: float = 0.0
    satisfaction: float = 0.0

    def update(self, text: str, analyzer: EmotionAnalyzer) -> None:
        """Update state based on *text* classified by *analyzer*.

        The excitement and satisfaction scores are adjusted in a small range
        [0.0, 1.0] to simulate an evolving emotional state.  Positive inputs
        increase the scores while negative inputs decrease them.
        """

        self.label = analyzer.analyze_text(text)
        if self.label == "positive":
            self.excitement = min(1.0, self.excitement + 0.1)
            self.satisfaction = min(1.0, self.satisfaction + 0.1)
        elif self.label == "negative":
            self.excitement = max(0.0, self.excitement - 0.1)
            self.satisfaction = max(0.0, self.satisfaction - 0.1)


def adjust_response_style(response: str, state: EmotionState) -> str:
    """Return *response* adapted to the current :class:`EmotionState`.

    The function adds a very small amount of stylistic variation based on the
    latest detected emotion.  While intentionally simple, this demonstrates how
    downstream components can modulate their behaviour based on emotional
    context.
    """

    if state.label == "positive":
        return f"{response} \U0001F60A"  # smiling face emoji
    if state.label == "negative":
        return f"I'm sorry to hear that. {response}"
    return response


__all__ = [
    "EmotionAnalyzer",
    "EmotionState",
    "adjust_response_style",
]
