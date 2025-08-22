from __future__ import annotations

from typing import Any, Optional

from pydantic import Field

from autogpt.core.agent.layered import LayeredAgent
from autogpt.core.configuration import (
    Configurable,
    SystemConfiguration,
    SystemSettings,
    UserConfigurable,
)


class GovernancePolicy(SystemConfiguration):
    """Settings that define governance rules for task routing."""

    allowed_task_types: list[str] = UserConfigurable(
        default_factory=list,
        description="List of task types permitted by the governance layer.",
    )


class GovernanceAgentSettings(SystemSettings):
    """System settings for the ``GovernanceAgent``."""

    policy: GovernancePolicy = Field(default_factory=GovernancePolicy)


class GovernanceAgent(LayeredAgent, Configurable[GovernanceAgentSettings]):
    """Top layer agent enforcing governance policy before delegating tasks."""

    default_settings = GovernanceAgentSettings(
        name="governance_agent",
        description="Routes tasks according to high level governance policy.",
    )

    def __init__(
        self,
        settings: GovernanceAgentSettings,
        next_layer: Optional[LayeredAgent] = None,
    ) -> None:
        super().__init__(next_layer=next_layer)
        self.settings = settings

    def route_task(self, task: Any, *args, **kwargs):
        """Route a task to the next layer if permitted by policy."""

        task_type = getattr(task, "type", getattr(task, "name", str(task)))
        allowed = self.settings.policy.allowed_task_types
        if allowed and task_type not in allowed:
            raise PermissionError(
                f"Task '{task_type}' is not permitted by the governance policy."
            )

        if self.next_layer is not None:
            return self.next_layer.route_task(task, *args, **kwargs)

        return task
