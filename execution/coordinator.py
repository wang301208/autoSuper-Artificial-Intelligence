"""Coordinator handling task distribution and status monitoring."""

from __future__ import annotations

from typing import Any, Dict

from events import EventBus
from events.coordination import TaskDispatchEvent, TaskStatus


class AgentCoordinator:
    """Listen for task status events and (re)dispatch tasks."""

    def __init__(self, event_bus: EventBus) -> None:
        self._bus = event_bus
        self._bus.subscribe("agent.status", self._on_status)

    def dispatch_task(
        self, task_id: str, payload: Dict[str, Any], agent_id: str | None = None
    ) -> None:
        event = TaskDispatchEvent(task_id=task_id, payload=payload, assigned_to=agent_id)
        self._bus.publish("task.dispatch", event.to_dict())

    def _on_status(self, event: Dict[str, Any]) -> None:
        status = event.get("status")
        if status == TaskStatus.COMPLETED.value:
            self._bus.publish("coordinator.task_completed", event)
        elif status == TaskStatus.FAILED.value:
            self._bus.publish("coordinator.task_failed", event)
