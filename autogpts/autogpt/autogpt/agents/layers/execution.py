from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from autogpt.core.ability import AbilityRegistry, SimpleAbilityRegistry
from autogpt.core.ability.schema import AbilityResult
from autogpt.core.agent.layered import LayeredAgent


class ExecutionAgent(LayeredAgent):
    """Layer responsible for executing abilities from a plan."""

    def __init__(
        self,
        ability_registry: AbilityRegistry | SimpleAbilityRegistry,
        next_layer: Optional[LayeredAgent] = None,
    ) -> None:
        super().__init__(next_layer=next_layer)
        self._ability_registry = ability_registry

    @classmethod
    def from_workspace(
        cls, workspace_path: Path, logger: Any
    ) -> "ExecutionAgent":  # pragma: no cover - simple passthrough
        raise NotImplementedError("ExecutionAgent does not support workspace loading")

    def __repr__(self) -> str:  # pragma: no cover - simple representation
        return f"ExecutionAgent(registry={self._ability_registry})"

    async def route_task(
        self, plan: dict[str, Any], *args: Any, **kwargs: Any
    ) -> AbilityResult:
        """Execute the ability described in ``plan`` and return the result."""

        ability_name = plan.get("next_ability")
        ability_args = plan.get("ability_arguments", {})
        try:
            result = await self._ability_registry.perform(ability_name, **ability_args)
        except Exception as err:
            if self.next_layer is not None:
                return await self.next_layer.route_task(err, *args, **kwargs)
            raise

        if self.next_layer is not None:
            return await self.next_layer.route_task(result, *args, **kwargs)
        return result
