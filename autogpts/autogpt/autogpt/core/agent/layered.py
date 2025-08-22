from __future__ import annotations

from typing import Any, Optional

from autogpt.core.agent.base import Agent


class LayeredAgent(Agent):
    """An agent capable of forwarding tasks to a subsequent layer."""

    next_layer: Optional["LayeredAgent"]

    def __init__(self, *args, next_layer: Optional["LayeredAgent"] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.next_layer = next_layer

    def route_task(self, task: Any, *args, **kwargs):
        """Route a task to the next layer by default."""
        if self.next_layer is not None:
            return self.next_layer.route_task(task, *args, **kwargs)
        raise NotImplementedError("No next layer to route task to.")
