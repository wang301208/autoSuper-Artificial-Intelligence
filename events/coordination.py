"""Standard event definitions for agent coordination."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Topics used for coordination events
TASK_PUBLISH_TOPIC = "coordination.task.publish"
STATUS_SYNC_TOPIC = "coordination.agent.status"
TASK_COMPLETE_TOPIC = "coordination.task.complete"
TASK_FAILED_TOPIC = "coordination.task.failed"


@dataclass
class TaskEvent:
    """Event emitted when a new task is published for agents."""

    task_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    target: Optional[str] = None


@dataclass
class StatusEvent:
    """Event used by agents to report their current status."""

    agent: str
    status: str
    progress: Optional[float] = None
    info: Dict[str, Any] = field(default_factory=dict)
