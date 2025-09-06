from __future__ import annotations

from typing import Dict


class SelfModel:
    """Estimate the agent's own state using environment predictions."""

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
