"""Agent coordination utilities."""

from __future__ import annotations

from typing import Any, Dict

from events import EventBus
from events.coordination import (
    STATUS_SYNC_TOPIC,
    TASK_COMPLETE_TOPIC,
    TASK_FAILED_TOPIC,
    TASK_PUBLISH_TOPIC,
    StatusEvent,
    TaskEvent,
)


class AgentCoordinator:
    """Listen for task events and coordinate work between agents."""

    def __init__(self, event_bus: EventBus) -> None:
        self._bus = event_bus
        self._statuses: Dict[str, Dict[str, Any]] = {}
        self._bus.subscribe(TASK_COMPLETE_TOPIC, self._on_task_complete)
        self._bus.subscribe(TASK_FAILED_TOPIC, self._on_task_failed)
        self._bus.subscribe(STATUS_SYNC_TOPIC, self._on_status)

    # ------------------------------------------------------------------
    def _on_task_complete(self, event: Dict[str, Any]) -> None:
        """Handle task completion by redistributing work."""
        self._bus.publish("coordination.reassign", event)

    def _on_task_failed(self, event: Dict[str, Any]) -> None:
        """Handle task failure by redistributing work."""
        self._bus.publish("coordination.reassign", event)

    def _on_status(self, event: Dict[str, Any]) -> None:
        """Track latest status reported by agents."""
        agent = event.get("agent")
        if agent:
            self._statuses[agent] = event

    # ------------------------------------------------------------------
    def publish_task(self, task: TaskEvent) -> None:
        self._bus.publish(TASK_PUBLISH_TOPIC, task.__dict__)

    def get_status(self, agent: str) -> Dict[str, Any] | None:
        return self._statuses.get(agent)
