from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Any, Optional

from .task_graph import TaskGraph


class Scheduler:
    """Dispatch tasks to the least busy agents based on resource usage."""

    def __init__(self, task_callback: Optional[Callable[[int], None]] = None) -> None:
        self._agents: Dict[str, Dict[str, float]] = {}
        self._task_counts: Dict[str, int] = {}
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
            self._agents.setdefault(name, {"cpu": 0.0, "memory": 0.0})
            self._task_counts.setdefault(name, 0)

    def remove_agent(self, name: str) -> None:
        """Remove an agent from scheduling."""
        with self._lock:
            self._agents.pop(name, None)
            self._task_counts.pop(name, None)

    def update_agent(self, name: str, cpu: float, memory: float) -> None:
        """Update utilization metrics for an agent."""
        with self._lock:
            if name in self._agents:
                self._agents[name]["cpu"] = cpu
                self._agents[name]["memory"] = memory

    @property
    def task_counts(self) -> Dict[str, int]:
        """Return a snapshot of tasks executed per agent."""
        with self._lock:
            return dict(self._task_counts)

    # ------------------------------------------------------------------
    def _pick_least_busy(self) -> str | None:
        with self._lock:
            if not self._agents:
                return None
            return min(
                self._agents.items(),
                key=lambda item: (item[1].get("cpu", 0.0), item[1].get("memory", 0.0)),
            )[0]

    # ------------------------------------------------------------------
    def submit(
        self, graph: TaskGraph, worker: Callable[[str, str], Any]
    ) -> Dict[str, Any]:
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
        in_progress: Dict[Any, str] = {}
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
                            results[task_id] = None
                            continue
                        with self._lock:
                            self._task_counts[agent] = (
                                self._task_counts.get(agent, 0) + 1
                            )
                        future = pool.submit(worker, agent, task.skill)
                        in_progress[future] = task_id
                    else:
                        results[task_id] = None
                if not in_progress:
                    continue
                done = next(as_completed(list(in_progress.keys())))
                tid = in_progress.pop(done)
                results[tid] = done.result()
                for dep in dependents.get(tid, []):
                    indegree[dep] -= 1
                    if indegree[dep] == 0:
                        ready.append(dep)
        if self._task_callback:
            self._task_callback(0)
        # Ensure deterministic order
        return {tid: results.get(tid) for tid in graph.execution_order()}


__all__ = ["Scheduler"]
