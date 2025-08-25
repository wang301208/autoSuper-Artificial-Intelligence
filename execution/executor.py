from __future__ import annotations

import ast
import re
import logging
from typing import Any, Dict, List

from capability.skill_library import SkillLibrary
from common.async_utils import run_async
from common.security import SAFE_BUILTINS, SkillSecurityError, _verify_skill
from .task_graph import TaskGraph
from .scheduler import Scheduler

logger = logging.getLogger(__name__)

class SkillExecutionError(RuntimeError):
    def __init__(self, skill: str, cause: str) -> None:
        super().__init__(f"Skill {skill} failed: {cause}")
        self.skill = skill
        self.cause = cause


class Executor:
    """Very small executor that decomposes a goal into skill tasks."""

    def __init__(self, skill_library: SkillLibrary, scheduler: Scheduler | None = None) -> None:
        self.skill_library = skill_library
        self.scheduler = scheduler or Scheduler()
        if not self.scheduler._agents:
            # Ensure at least one local agent for execution
            self.scheduler.add_agent("local")

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
        return self.scheduler.submit(
            graph, lambda agent, skill: run_async(self._call_skill_async(agent, skill))
        )

    def _call_skill(self, agent: str, name: str) -> Any:
        """Synchronously execute ``name`` skill for ``agent``."""

        return run_async(self._call_skill_async(agent, name))

    async def _call_skill_async(self, agent: str, name: str) -> Any:
        """Execute ``name`` skill for ``agent``.

        The basic executor only supports a local ``SkillLibrary`` and therefore
        ignores the ``agent`` argument, but the parameter allows alternative
        implementations to route tasks to remote agents or specialized
        resources.
        """
        code, meta = await self.skill_library.get_skill(name)
        try:
            _verify_skill(name, code, meta)
        except SkillSecurityError as err:
            logger.error("Security violation for skill %s: %s", name, err.cause)
            raise
        namespace: Dict[str, Any] = {}
        parsed = ast.parse(code, mode="exec")
        exec(compile(parsed, filename=name, mode="exec"), {"__builtins__": SAFE_BUILTINS}, namespace)
        func = namespace.get(name)
        if not callable(func):
            err = SkillExecutionError(name, f"did not define a callable {name}()")
            logger.error(str(err))
            raise err
        try:
            return func()
        except Exception as err:  # noqa: BLE001
            logger.exception("Error executing skill %s: %s", name, err)
            raise SkillExecutionError(name, str(err)) from err
