from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, Optional, List


class TaskStatus(str, Enum):
    """Enumeration of possible task states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskDispatchEvent:
    """Event published by the coordinator to assign a task to an agent."""

    task_id: str
    payload: Dict[str, Any]
    assigned_to: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskStatusEvent:
    """Event emitted by an agent to report the status of a task."""

    agent_id: str
    task_id: str
    status: TaskStatus
    detail: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass
class IterationEvent:
    """Event representing a reflexive iteration step."""

    iteration: int
    candidates: List[str]
    selected: str
    scores: Dict[str, float] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def render_iteration_timeline(events: List[IterationEvent]) -> str:
    """Return a simple textual visualization of iteration events."""

    lines = []
    for ev in events:
        score_part = ""
        if ev.scores:
            formatted = ", ".join(
                f"{name}:{score:.2f}" for name, score in ev.scores.items()
            )
            score_part = f" [{formatted}]"
        lines.append(f"{ev.iteration}: {ev.selected}{score_part}")
    return "\n".join(lines)
