from __future__ import annotations

from typing import Any, Dict, List, Tuple

from backend.reflection.reflection import ReflectionModule
from backend.memory.long_term import LongTermMemory


class SelfModel:
    """Estimate the agent's own state using environment predictions."""

    def __init__(self, memory: LongTermMemory | None = None) -> None:
        self._reflection = ReflectionModule()
        self._history: List[str] = []
        self._memory = memory
        self._last_summary: str | None = None
        self._self_state: Dict[str, Any] = {
            "goals": [],
            "capabilities": [],
            "mood": "neutral",
        }

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

    def update_state(self, events: List[str]) -> None:
        """Update the internal ``self_state`` based on ``events``."""

        for event in events:
            e = event.lower()
            if "goal" in e and ":" in event:
                self._self_state["goals"].append(event.split(":", 1)[1].strip())
            if "capability" in e and ":" in event:
                self._self_state["capabilities"].append(event.split(":", 1)[1].strip())
            if any(word in e for word in ("error", "fail")):
                self._self_state["mood"] = "frustrated"
            elif any(word in e for word in ("success", "completed", "done")):
                self._self_state["mood"] = "satisfied"

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

        self.update_state([last_action])
        metrics = self.estimate(data, env_pred)
        base = (
            f"cpu={metrics['cpu']:.2f}, memory={metrics['memory']:.2f}; "
            f"last_action={last_action}"
        )
        evaluation, revised = self._reflection.reflect(base)
        summary = f"{evaluation} | {revised}"

        narrative_base = (
            f"mood={self._self_state['mood']}; goals="
            f"{', '.join(self._self_state['goals']) or 'none'}; capabilities="
            f"{', '.join(self._self_state['capabilities']) or 'none'}"
        )
        n_eval, n_revised = self._reflection.reflect(narrative_base)
        narrative = f"{n_eval} | {n_revised}"
        summary = f"{summary} | {narrative}"

        self._history.append(summary)
        self._history = self._history[-5:]
        self._last_summary = summary
        if self._memory:
            self._memory.add("self_awareness", summary)
            self._memory.add("self_narrative", narrative)
        return metrics, summary

    # ------------------------------------------------------------------
    def introspect(
        self, data: Dict[str, float], env_pred: Dict[str, float], last_action: str
    ) -> Dict[str, float | str]:
        """Return CPU/memory estimates together with a reflection summary."""

        metrics, summary = self.assess_state(data, env_pred, last_action)
        return {"cpu": metrics["cpu"], "memory": metrics["memory"], "summary": summary}

    @property
    def history(self) -> List[str]:
        return list(self._history)

    @property
    def last_summary(self) -> str | None:
        """Return the most recent introspection summary."""

        return self._last_summary
