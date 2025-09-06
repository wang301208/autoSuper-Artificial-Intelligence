from __future__ import annotations

from typing import Dict, List, Tuple

from backend.reflection.reflection import ReflectionModule
from backend.memory.long_term import LongTermMemory


class SelfModel:
    """Estimate the agent's own state using environment predictions."""

    def __init__(self, memory: LongTermMemory | None = None) -> None:
        self._reflection = ReflectionModule()
        self._history: List[str] = []
        self._memory = memory

    def estimate(self, data: Dict[str, float], env_pred: Dict[str, float]) -> Dict[str, float]:
        """Return corrected CPU and memory predictions.

        The correction subtracts 10% of the average environment load from the
        agent's current usage as a simplistic feedback mechanism.
        """

        adjustment_cpu = env_pred.get("avg_cpu", 0.0) * 0.1
        adjustment_mem = env_pred.get("avg_memory", 0.0) * 0.1
        return {
            "cpu": max(data.get("cpu", 0.0) - adjustment_cpu, 0.0),
            "memory": max(data.get("memory", 0.0) - adjustment_mem, 0.0),
        }

    def assess_state(
        self,
        data: Dict[str, float],
        env_pred: Dict[str, float],
        last_action: str,
    ) -> Tuple[Dict[str, float], str]:
        """Return corrected metrics and an introspective summary.

        The summary is generated using :class:`ReflectionModule` and recent
        results are stored for future reference.
        """

        metrics = self.estimate(data, env_pred)
        base = (
            f"cpu={metrics['cpu']:.2f}, memory={metrics['memory']:.2f}; "
            f"last_action={last_action}"
        )
        evaluation, revised = self._reflection.reflect(base)
        summary = f"{evaluation} | {revised}"
        self._history.append(summary)
        self._history = self._history[-5:]
        if self._memory:
            self._memory.add("self_awareness", summary)
        return metrics, summary

    @property
    def history(self) -> List[str]:
        return list(self._history)
