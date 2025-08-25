from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

from .task_graph import TaskGraph


class Scheduler:
    """Dispatch tasks to the least busy agents based on resource usage."""

    def __init__(
        self,
        task_callback: Optional[Callable[[int], None]] = None,
        *,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        env_weights = {
            "cpu": float(os.getenv("SCHEDULER_CPU_WEIGHT", 1.0)),
            "memory": float(os.getenv("SCHEDULER_MEMORY_WEIGHT", 1.0)),
            "tasks": float(os.getenv("SCHEDULER_TASK_WEIGHT", 1.0)),
        }
        if weights:
            env_weights.update(weights)
        self._weights = env_weights
        self._agents: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()
        self._task_callback = task_callback

    def set_task_callback(self, cb: Callable[[int], None]) -> None:
        """Set a callback to be notified when task counts change."""
        self._task_callback = cb

    # ------------------------------------------------------------------
    # Agent management API
    # ------------------------------------------------------------------
    def add_agent(self, name: str) -> None:
        """Register a new agent with default utilization."""
        with self._lock:
            self._agents.setdefault(
                name, {"cpu": 0.0, "memory": 0.0, "tasks": 0.0}
            )

    def remove_agent(self, name: str) -> None:
        """Remove an agent from scheduling."""
        with self._lock:
            self._agents.pop(name, None)

    def update_agent(self, name: str, cpu: float, memory: float) -> None:
        """Update utilization metrics for an agent."""
        with self._lock:
            if name in self._agents:
                self._agents[name]["cpu"] = cpu
                self._agents[name]["memory"] = memory

    # ------------------------------------------------------------------
    def _pick_least_busy(self) -> str | None:
        with self._lock:
            if not self._agents:
                return None

            def score(item: tuple[str, Dict[str, float]]) -> float:
                metrics = item[1]
                return (
                    self._weights["cpu"] * metrics.get("cpu", 0.0)
                    + self._weights["memory"] * metrics.get("memory", 0.0)
                    + self._weights["tasks"] * metrics.get("tasks", 0.0)
                )

            return min(self._agents.items(), key=score)[0]

    # ------------------------------------------------------------------
    def submit(self, graph: TaskGraph, worker: Callable[[str], Any]) -> Dict[str, Any]:
        """Schedule tasks on available agents and execute them in parallel."""
        # Build dependency counts and reverse edges
        indegree: Dict[str, int] = {}
        dependents: Dict[str, List[str]] = {}
        for task_id, task in graph.tasks.items():
            indegree[task_id] = len(task.dependencies)
            for dep in task.dependencies:
                dependents.setdefault(dep, []).append(task_id)

        ready = [tid for tid, deg in indegree.items() if deg == 0]
        if self._task_callback:
            self._task_callback(len(ready))
        results: Dict[str, Any] = {}
        in_progress: Dict[Any, tuple[str, str]] = {}
        max_workers = max(len(self._agents), 1)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            while ready or in_progress:
                # Fill available worker slots
                while ready and len(in_progress) < max_workers:
                    task_id = ready.pop(0)
                    task = graph.tasks[task_id]
                    if task.skill:
                        agent = self._pick_least_busy()
                        if agent is None:
                            continue
                        future = pool.submit(worker, task.skill)
                        in_progress[future] = (task_id, agent)
                        with self._lock:
                            self._agents[agent]["tasks"] += 1
                    else:
                        results[task_id] = None
                if not in_progress:
                    continue
                done = next(as_completed(list(in_progress.keys())))
                tid, agent = in_progress.pop(done)
                results[tid] = done.result()
                with self._lock:
                    self._agents[agent]["tasks"] -= 1
                for dep in dependents.get(tid, []):
                    indegree[dep] -= 1
                    if indegree[dep] == 0:
                        ready.append(dep)
        if self._task_callback:
            self._task_callback(0)
        # Ensure deterministic order
        return {tid: results.get(tid) for tid in graph.execution_order()}


__all__ = ["Scheduler"]
