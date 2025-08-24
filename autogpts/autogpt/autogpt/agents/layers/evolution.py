from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable, Optional

from ...core.agent.simple import PerformanceEvaluator
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

        data_dir = Path(__file__).resolve().parents[3] / "data"
        data_dir.mkdir(exist_ok=True)
        self._q_table_path = data_dir / "q_table.json"
        self._q_table: dict[str, dict[str, float]] = {}
        self._learning_rate = 0.1
        self._discount_factor = 0.9
        self._exploration_rate = 0.1
        self._performance = PerformanceEvaluator()
        self._last_state_action: Optional[tuple[str, str]] = None

        self._load_q_table()

    def _load_q_table(self) -> None:
        if self._q_table_path.exists():
            try:
                self._q_table = json.loads(self._q_table_path.read_text())
            except Exception:
                self._q_table = {}

    def _save_q_table(self) -> None:
        try:
            self._q_table_path.write_text(json.dumps(self._q_table))
        except Exception:
            pass

    def _select_ability(self, state: str, abilities: list[Any]) -> Any:
        abilities_names = [getattr(a, "name", str(a)) for a in abilities]
        if random.random() < self._exploration_rate:
            idx = random.randrange(len(abilities))
            return abilities[idx]
        state_values = self._q_table.get(state, {})
        best_ability = max(
            abilities_names,
            key=lambda a: state_values.get(a, 0.0),
            default=abilities_names[0] if abilities_names else None,
        )
        for ability in abilities:
            if getattr(ability, "name", str(ability)) == best_ability:
                return ability
        return abilities[0]

    def record_feedback(self, ability_name: str, result: Any) -> None:
        if self._last_state_action is None:
            return
        state, action = self._last_state_action
        reward = self._performance.score(result, cost=0.0, duration=0.0)
        state_values = self._q_table.setdefault(state, {})
        old_value = state_values.get(action, 0.0)
        new_value = old_value + self._learning_rate * (reward - old_value)
        state_values[action] = new_value
        self._save_q_table()

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

        chosen_task = filtered_tasks[0] if filtered_tasks else None
        if chosen_task and filtered_abilities:
            state = getattr(chosen_task, "id", getattr(chosen_task, "name", str(chosen_task)))
            ability = self._select_ability(state, filtered_abilities)
            ability_name = getattr(ability, "name", str(ability))
            self._last_state_action = (state, ability_name)
            filtered_tasks = [chosen_task]
            filtered_abilities = [ability]

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
