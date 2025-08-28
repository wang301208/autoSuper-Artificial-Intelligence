from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, Optional


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
