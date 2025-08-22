from __future__ import annotations

from typing import Any, Iterable, Optional

from autogpt.core.agent.layered import LayeredAgent
from autogpt.core.memory import Memory
from autogpt.core.planning import SimplePlanner


class EvolutionAgent(LayeredAgent):
    """Layer that adapts tasks and abilities based on past outcomes.

    This agent sits between planning and ability execution layers. It inspects
    historical task results stored in memory or planning subsystems and filters
    out tasks or abilities that have repeatedly failed. The remaining tasks are
    delegated to the next layer for ability selection or execution.
    """

    def __init__(
        self,
        memory: Optional[Memory] = None,
        planning: Optional[SimplePlanner] = None,
        next_layer: Optional[LayeredAgent] = None,
    ) -> None:
        super().__init__(next_layer=next_layer)
        self._memory = memory
        self._planning = planning

    async def determine_next_ability(
        self,
        task_queue: Iterable[Any],
        ability_list: Iterable[Any],
        *args: Any,
        **kwargs: Any,
    ):
        """Evaluate history and forward filtered tasks to the next layer."""

        history: list[Any] = []
        if self._memory is not None:
            try:
                history.extend(self._memory.get() or [])
            except Exception:
                pass
        if self._planning is not None and hasattr(
            self._planning, "get_completed_tasks"
        ):
            try:
                history.extend(self._planning.get_completed_tasks())
            except Exception:
                pass

        task_queue = list(task_queue)
        ability_list = list(ability_list)

        def task_failed(task: Any) -> bool:
            identifier = getattr(task, "id", getattr(task, "name", str(task)))
            return any(
                identifier in str(item) and "FAIL" in str(item)
                for item in history
            )

        def ability_failed(name: str) -> bool:
            return any(
                name in str(item) and "ABILITY_FAIL" in str(item)
                for item in history
            )

        filtered_tasks = [t for t in task_queue if not task_failed(t)]
        filtered_abilities = [
            a for a in ability_list if not ability_failed(getattr(a, "name", str(a)))
        ]

        if self._planning is not None and hasattr(self._planning, "update_tasks"):
            try:
                self._planning.update_tasks(filtered_tasks)
            except Exception:
                pass

        if self._memory is not None and hasattr(self._memory, "add"):
            if len(filtered_tasks) < len(task_queue):
                self._memory.add("EvolutionAgent: removed failed tasks from queue.")

        if self.next_layer is not None:
            return await self.next_layer.determine_next_ability(
                filtered_tasks, filtered_abilities, *args, **kwargs
            )
        return filtered_tasks, filtered_abilities
