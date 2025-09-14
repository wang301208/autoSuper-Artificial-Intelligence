"""Simple hexagram engine that can utilize time context."""

from __future__ import annotations

from typing import Optional

from .time_context import TimeContext


class HexagramEngine:
    """Generate basic I Ching interpretations with optional time context."""

    def interpret(self, hexagram: str, context: Optional[TimeContext] = None) -> str:
        """Return an interpretation string for the given hexagram.

        If ``context`` is provided, the solar term and Chinese hour are included in
        the interpretation to illustrate how timing can affect divination results.
        """

        explanation = f"Interpretation for {hexagram}"
        if context:
            explanation += (
                f" during {context.solar_term} {context.chinese_hour} hour"
            )
        return explanation
