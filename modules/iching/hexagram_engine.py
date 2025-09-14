"""Engine for generating modern I Ching interpretations."""
from __future__ import annotations

from .ai_interpreter import AIEnhancedInterpreter


class HexagramEngine:
    """Generate interpretations for hexagrams in given contexts."""

    def __init__(self, interpreter: AIEnhancedInterpreter | None = None):
        self.interpreter = interpreter or AIEnhancedInterpreter()

    def get_interpretation(self, hexagram: str, context: str | None = None) -> str:
        """Return a context-aware interpretation for *hexagram*."""
        return self.interpreter.interpret(hexagram, context)
