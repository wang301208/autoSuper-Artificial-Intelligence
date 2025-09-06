from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SelfState:
    """Container for the agent's internal self state."""

    current_goal: str = ""
    memory_pointer: int = 0
    action_history: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "current_goal": self.current_goal,
            "memory_pointer": self.memory_pointer,
            "recent_actions": self.action_history[-5:],
        }


class GlobalWorkspace:
    """Simple global workspace for coordinating agent subsystems."""

    def __init__(self) -> None:
        self.state = SelfState()

    def update_state(
        self,
        goal: str | None = None,
        memory_pointer: int | None = None,
        action: str | None = None,
    ) -> None:
        if goal is not None:
            self.state.current_goal = goal
        if memory_pointer is not None:
            self.state.memory_pointer = memory_pointer
        if action is not None:
            self.state.action_history.append(action)

    def get_context(self) -> Dict[str, Any]:
        """Return a dictionary representation of the current self state."""
        return self.state.as_dict()

    def reflect(self, predicted_action: str, actual_action: str) -> None:
        """Very small reflection mechanism comparing expected and actual actions."""
        if predicted_action != actual_action:
            self.state.action_history.append(
                f"reflection: expected {predicted_action} but executed {actual_action}"
            )
