"""Founder agent that observes system metrics and suggests improvements."""

from __future__ import annotations

import os
from typing import Any, Dict

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None

from . import Agent
from .genesis_team import GenesisTeamManager
from .ml_model import ResourceModel
from .self_improvement import SelfImprovement


class Founder(Agent):
    """Agent that reads system performance metrics and outputs suggestions."""

    def __init__(self) -> None:
        self.model = ResourceModel()

    def perform(self) -> str:
        metrics = self._collect_metrics()
        suggestions = self._generate_suggestions(metrics)
        return "\n".join(suggestions)

    def _collect_metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        if psutil is not None:
            metrics["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            metrics["memory_percent"] = psutil.virtual_memory().percent
        elif hasattr(os, "getloadavg"):
            metrics["load_avg"] = os.getloadavg()
        return metrics

    def _generate_suggestions(self, metrics: Dict[str, Any]) -> list[str]:
        suggestions: list[str] = []
        predictions = self.model.predict_next()
        cpu = predictions.get("cpu_percent") or metrics.get("cpu_percent")
        mem = predictions.get("memory_percent") or metrics.get("memory_percent")
        if cpu is not None and cpu > 80:
            suggestions.append("CPU usage high; consider distributing tasks.")
        if mem is not None and mem > 80:
            suggestions.append("Memory usage high; investigate memory leaks.")
        if not suggestions:
            suggestions.append("System resources stable; maintain current structure.")
        return suggestions

    def plan_tool_updates(self) -> str:
        """Run the Genesis team and self-improvement routine."""

        manager = GenesisTeamManager()
        logs = manager.run()

        improver = SelfImprovement()
        results = improver.run()

        summary = [f"{agent}: {output}" for agent, output in logs.items()]
        if results["suggestions"]:
            summary.append("Self-Improvement Suggestions:")
            summary.extend(results["suggestions"])
        if results["actions"]:
            summary.append("Triggered Actions:")
            summary.extend(results["actions"])
        return "\n".join(summary)
