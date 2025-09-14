from __future__ import annotations

"""Simple engine for working with I Ching hexagrams.

This module provides :class:`HexagramEngine` that can predict future
hexagrams based on changing lines (爻变) or simple time progression.
The engine records transformation paths and basic trend metrics and
returns a textual report summarising the trend.
"""

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Optional


@dataclass
class TransformationResult:
    """Result returned by :meth:`HexagramEngine.predict_transformations`.

    Attributes
    ----------
    path:
        Sequence of hexagrams encountered during the prediction.  Each
        hexagram is represented as a tuple of six integers (0 for yin,
        1 for yang).
    yang_counts:
        Number of yang lines in each hexagram along ``path``.
    trend:
        ``"increasing"``, ``"decreasing"`` or ``"stable"`` depending on how
        the number of yang lines changed from the initial to the final
        hexagram.
    report:
        Human readable textual report describing the trend.
    """

    path: List[Tuple[int, ...]]
    yang_counts: List[int]
    trend: str
    report: str


class HexagramEngine:
    """Engine for predicting I Ching hexagram transformations.

    Parameters
    ----------
    initial:
        Initial hexagram represented as a sequence of six integers where
        ``0`` denotes a yin line and ``1`` denotes a yang line.  Index ``0``
        corresponds to the bottom line.
    """

    def __init__(self, initial: Sequence[int]):
        if len(initial) != 6:
            raise ValueError("Hexagram must consist of 6 lines")
        if any(line not in (0, 1) for line in initial):
            raise ValueError("Lines must be 0 (yin) or 1 (yang)")
        self.current: Tuple[int, ...] = tuple(int(line) for line in initial)
        self.history: List[Tuple[int, ...]] = [self.current]

    def transform(self, indices: Iterable[int]) -> Tuple[int, ...]:
        """Apply a single transformation by flipping lines at ``indices``."""

        lines = list(self.current)
        for idx in indices:
            if idx < 0 or idx >= 6:
                raise IndexError("line index out of range")
            lines[idx] = 1 - lines[idx]
        self.current = tuple(lines)
        self.history.append(self.current)
        return self.current

    def predict_transformations(
        self,
        changes: Optional[Sequence[Iterable[int]]] = None,
        *,
        steps: int = 0,
    ) -> TransformationResult:
        """Predict future hexagrams based on line changes or time steps.

        Only one of ``changes`` or ``steps`` should be provided.  When
        ``changes`` is given it should be a sequence where each element is an
        iterable of line indices (0-based, bottom to top) to flip for that
        step.  When ``steps`` is provided the engine performs that many steps
        by sequentially flipping one line at a time from bottom to top.
        """

        if changes and steps:
            raise ValueError("Specify either 'changes' or 'steps', not both")

        # Start a new prediction history beginning with the current hexagram
        self.history = [self.current]

        if changes:
            for step_changes in changes:
                self.transform(step_changes)
        else:
            for step in range(steps):
                # Simple time-based rule: flip one line in sequence
                self.transform([step % 6])

        yang_counts = [sum(h) for h in self.history]
        trend_direction = "stable"
        if yang_counts[-1] > yang_counts[0]:
            trend_direction = "increasing"
        elif yang_counts[-1] < yang_counts[0]:
            trend_direction = "decreasing"

        report = (
            f"Yang lines changed from {yang_counts[0]} to {yang_counts[-1]} over "
            f"{len(self.history) - 1} steps ({trend_direction})."
        )

        return TransformationResult(
            path=self.history.copy(),
            yang_counts=yang_counts,
            trend=trend_direction,
            report=report,
        )

    def get_trend_report(self) -> str:
        """Return a textual report for the last prediction."""

        if not self.history:
            return "No prediction performed."
        yang_counts = [sum(h) for h in self.history]
        trend_direction = "stable"
        if yang_counts[-1] > yang_counts[0]:
            trend_direction = "increasing"
        elif yang_counts[-1] < yang_counts[0]:
            trend_direction = "decreasing"
        return (
            f"Yang lines changed from {yang_counts[0]} to {yang_counts[-1]} over "
            f"{len(self.history) - 1} steps ({trend_direction})."
        )
