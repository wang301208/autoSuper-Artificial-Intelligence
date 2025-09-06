from __future__ import annotations

from typing import Dict


class WorldModel:
    """Simple environment predictor.

    Computes average CPU and memory usage across agents.
    """

    def predict(self, resources: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        if not resources:
            return {"avg_cpu": 0.0, "avg_memory": 0.0}
        total_cpu = sum(r.get("cpu", 0.0) for r in resources.values())
        total_mem = sum(r.get("memory", 0.0) for r in resources.values())
        count = len(resources)
        return {"avg_cpu": total_cpu / count, "avg_memory": total_mem / count}
