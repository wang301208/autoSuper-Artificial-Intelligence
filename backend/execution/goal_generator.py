"""Goal generation module.

This simple component proposes follow-up goals based on past outcomes stored
in long-term memory.  Each outcome is reflected upon using
:class:`ReflectionModule` to surface potential improvements, which are then
returned as new high-level goals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..memory import LongTermMemory
from ..reflection import ReflectionModule


@dataclass
class GoalGenerator:
    """Generate new goals from past outcomes."""

    reflection: ReflectionModule
    memory: LongTermMemory

    def __init__(
        self,
        reflection: Optional[ReflectionModule] = None,
        memory: Optional[LongTermMemory] = None,
    ) -> None:
        self.reflection = reflection or ReflectionModule()
        self.memory = memory or LongTermMemory(":memory:")

    def generate(self) -> Optional[str]:
        """Propose a new goal based on the most recent outcome.

        If no outcomes are stored, ``None`` is returned.
        """
        outcomes = list(self.memory.get("outcome"))
        if not outcomes:
            return None
        last = outcomes[-1]
        _, revised = self.reflection.reflect(last)
        return f"Build on: {revised}"


__all__ = ["GoalGenerator"]
