from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

from capability.skill_library import SkillLibrary
from .task_graph import TaskGraph


class Executor:
    """Very small executor that decomposes a goal into skill tasks."""

    def __init__(self, skill_library: SkillLibrary) -> None:
        self.skill_library = skill_library

    # Goal decomposition
    def decompose_goal(self, goal: str) -> TaskGraph:
        """Split a goal string into sequential skill tasks.

        The goal is split on the words 'then' or 'and'. Each resulting token is
        treated as a skill name if it matches a skill in the library. Subsequent
        tasks depend on the previous task, forming a simple chain.
        """

        graph = TaskGraph()
        tokens = [t.strip() for t in re.split(r"then|and", goal) if t.strip()]
        previous: str | None = None
        available = set(self.skill_library.list_skills())
        for token in tokens:
            if token in available:
                deps: List[str] = [previous] if previous else []
                graph.add_task(
                    token,
                    description=f"Execute {token}",
                    skill=token,
                    dependencies=deps,
                )
                previous = token
        return graph

    # Task scheduling and execution
    def execute(self, goal: str) -> Dict[str, Any]:
        graph = self.decompose_goal(goal)
        results: Dict[str, Any] = {}
        for task_id in graph.execution_order():
            task = graph.tasks[task_id]
            if task.skill:
                results[task_id] = self._call_skill(task.skill)
        return results

    def _call_skill(self, name: str) -> Any:
        code, _ = self.skill_library.get_skill(name)
        namespace: Dict[str, Any] = {}
        exec(code, namespace)
        func = namespace.get(name)
        if callable(func):
            return func()
        raise ValueError(f"Skill {name} did not define a callable {name}()")
