from __future__ import annotations

import heapq
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
        # Track total completed tasks for fairness verification
        self._task_counts: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._task_callback = task_callback
        # Heap of (score, revision, name) for O(log n) selection
        self._heap: List[tuple[float, int, str]] = []
        self._revisions: Dict[str, int] = {}

    def set_task_callback(self, cb: Callable[[int], None]) -> None:
        """Set a callback to be notified when task counts change."""
        self._task_callback = cb

    # ------------------------------------------------------------------
    # Agent management API
    # ------------------------------------------------------------------
    def add_agent(self, name: str) -> None:
        """Register a new agent with default utilization."""
        with self._lock:
            if name in self._agents:
                return
            self._agents[name] = {"cpu": 0.0, "memory": 0.0, "tasks": 0.0}
            self._task_counts[name] = 0
            self._push(name)

    def remove_agent(self, name: str) -> None:
        """Remove an agent from scheduling."""
        with self._lock:
            self._agents.pop(name, None)
            self._task_counts.pop(name, None)
            self._revisions.pop(name, None)

    def update_agent(self, name: str, cpu: float, memory: float) -> None:
        """Update utilization metrics for an agent."""
        with self._lock:
            if name in self._agents:
                self._agents[name]["cpu"] = cpu
                self._agents[name]["memory"] = memory
                self._push(name)

    def _score(self, name: str) -> float:
        metrics = self._agents[name]
        return (
            self._weights["cpu"] * metrics.get("cpu", 0.0)
            + self._weights["memory"] * metrics.get("memory", 0.0)
            + self._weights["tasks"] * metrics.get("tasks", 0.0)
        )

    def _push(self, name: str) -> None:
        rev = self._revisions.get(name, 0) + 1
        self._revisions[name] = rev
        heapq.heappush(self._heap, (self._score(name), rev, name))

    def _update_tasks(self, name: str, delta: float) -> None:
        self._agents[name]["tasks"] += delta
        self._push(name)

    # ------------------------------------------------------------------
    def _pick_least_busy(self) -> str | None:
        with self._lock:
            while self._heap:
                score, rev, name = heapq.heappop(self._heap)
                if name not in self._agents:
                    continue
                if self._revisions.get(name) != rev:
                    continue
                return name
            return None

    # ------------------------------------------------------------------
    def submit(
        self, graph: TaskGraph, worker: Callable[[str, str], Any]
    ) -> Dict[str, Any]:
        """Schedule tasks on available agents and execute them in parallel.

        The ``worker`` callable receives the selected ``agent`` name and the
        ``skill`` associated with each task, allowing downstream consumers to
        route execution appropriately.
        """
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
                        future = pool.submit(worker, agent, task.skill)
                        in_progress[future] = (task_id, agent)
                        with self._lock:
                            self._update_tasks(agent, 1)
                    else:
                        results[task_id] = None
                if not in_progress:
                    continue
                done = next(as_completed(list(in_progress.keys())))
                tid, agent = in_progress.pop(done)
                results[tid] = done.result()
                with self._lock:
                    self._update_tasks(agent, -1)
                    self._task_counts[agent] += 1
                for dep in dependents.get(tid, []):
                    indegree[dep] -= 1
                    if indegree[dep] == 0:
                        ready.append(dep)
        if self._task_callback:
            self._task_callback(0)
        # Ensure deterministic order
        return {tid: results.get(tid) for tid in graph.execution_order()}

    # ------------------------------------------------------------------
    def task_counts(self) -> Dict[str, int]:
        """Return a snapshot of total completed tasks per agent."""
        with self._lock:
            return dict(self._task_counts)


__all__ = ["Scheduler"]
