from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from autogpt.core.ability import AbilityRegistry, SimpleAbilityRegistry
from autogpt.core.ability.schema import AbilityResult
from autogpt.core.agent.layered import LayeredAgent


class CapabilityAgent(LayeredAgent):
    """Layer that selects abilities for a task and delegates execution.

    This layer inspects the available abilities provided by an ability registry
    and chooses the one that best matches the incoming task. The selected ability
    is wrapped into an execution plan and routed to the next layer, typically an
    execution layer responsible for actually running the ability. After
    execution, performance data is optionally forwarded to a feedback handler so
    that an evolution layer can learn from the outcome.
    """

    def __init__(
        self,
        ability_registry: AbilityRegistry | SimpleAbilityRegistry,
        next_layer: Optional[LayeredAgent] = None,
        feedback_handler: Optional[Callable[[str, AbilityResult], None]] = None,
    ) -> None:
        super().__init__(next_layer=next_layer)
        self._ability_registry = ability_registry
        self._feedback_handler = feedback_handler

    @classmethod
    def from_workspace(
        cls, workspace_path: Path, logger: Any
    ) -> "CapabilityAgent":  # pragma: no cover - simple passthrough
        raise NotImplementedError("CapabilityAgent does not support workspace loading")

    def __repr__(self) -> str:  # pragma: no cover - simple representation
        return f"CapabilityAgent(registry={self._ability_registry})"

    async def determine_next_ability(
        self, task: Any, *args: Any, **kwargs: Any
    ) -> tuple[dict[str, Any], AbilityResult]:
        """Select the best ability for ``task`` and delegate its execution."""

        ability_names = self._ability_registry.list_abilities()
        selected_ability = self._select_ability(task, ability_names)

        plan = {"next_ability": selected_ability, "ability_arguments": kwargs}

        if self.next_layer is not None:
            result = await self.next_layer.route_task(plan, *args, **kwargs)
        else:
            result = await self._ability_registry.perform(
                selected_ability, **plan["ability_arguments"]
            )

        if self._feedback_handler is not None:
            self._feedback_handler(selected_ability, result)

        return plan, result

    def _select_ability(
        self, task: Any, ability_names: Iterable[str]
    ) -> str:
        """Pick an ability that best matches ``task`` from ``ability_names``."""

        task_type = getattr(task, "type", getattr(task, "name", str(task)))
        for ability in ability_names:
            if task_type and task_type.lower() in ability.lower():
                return ability
        return next(iter(ability_names), "")

    def record_feedback(self, ability_name: str, result: AbilityResult) -> None:
        """Manually record feedback about an ability's performance."""

        if self._feedback_handler is not None:
            self._feedback_handler(ability_name, result)
