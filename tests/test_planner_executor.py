"""Tests for planning and execution helpers."""

from __future__ import annotations

import hashlib

from backend.reasoning import MultiHopAssociator
from backend.reflection import ReflectionModule
import importlib.util
from pathlib import Path


def _load(name: str, path: str):
    abs_path = Path(__file__).resolve().parent.parent / path
    spec = importlib.util.spec_from_file_location(name, str(abs_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    import sys

    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ps = _load("backend.creative_engine.problem_solver", "backend/creative_engine/problem_solver.py")
import types, sys

backend_ce = types.ModuleType("backend.creative_engine")
backend_ce.problem_solver = ps
sys.modules["backend.creative_engine"] = backend_ce
sys.modules["backend.creative_engine.problem_solver"] = ps
backend_execution = types.ModuleType("backend.execution")
backend_execution.__path__ = []
sys.modules["backend.execution"] = backend_execution

# Stub modules required by executor
errors_module = types.ModuleType("autogpts.autogpt.autogpt.core.errors")
errors_module.SkillExecutionError = type("SkillExecutionError", (Exception,), {})
errors_module.SkillSecurityError = type("SkillSecurityError", (Exception,), {})
sys.modules["autogpts"] = types.ModuleType("autogpts")
sys.modules["autogpts.autogpt"] = types.ModuleType("autogpts.autogpt")
sys.modules["autogpts.autogpt.autogpt"] = types.ModuleType("autogpts.autogpt.autogpt")
sys.modules["autogpts.autogpt.autogpt.core"] = types.ModuleType(
    "autogpts.autogpt.autogpt.core"
)
sys.modules["autogpts.autogpt.autogpt.core.errors"] = errors_module
skill_lib_module = types.ModuleType("capability.skill_library")
skill_lib_module.SkillLibrary = object
sys.modules["capability"] = types.ModuleType("capability")
sys.modules["capability.skill_library"] = skill_lib_module
async_utils = types.ModuleType("common.async_utils")

def _run_async(coro):
    import asyncio

    return asyncio.run(coro)

async_utils.run_async = _run_async
sys.modules["common"] = types.ModuleType("common")
sys.modules["common.async_utils"] = async_utils

task_graph_mod = _load("backend.execution.task_graph", "backend/execution/task_graph.py")
scheduler_mod = _load("backend.execution.scheduler", "backend/execution/scheduler.py")
planner_mod = _load("backend.execution.planner", "backend/execution/planner.py")
executor_mod = _load("backend.execution.executor", "backend/execution/executor.py")

DivergentConvergentSolver = ps.DivergentConvergentSolver
Planner = planner_mod.Planner
Executor = executor_mod.Executor


class DummySkillLibrary:
    def __init__(self) -> None:
        self.skills = {}
        self.add_skill("plan_a", "def plan_a():\n    return 'A'")
        self.add_skill("plan_b", "def plan_b():\n    return 'B'")

    def add_skill(self, name: str, code: str) -> None:
        sig = hashlib.sha256(code.encode("utf-8")).hexdigest()
        self.skills[name] = (code, {"signature": sig})

    def list_skills(self):  # pragma: no cover - trivial
        return list(self.skills.keys())

    async def get_skill(self, name: str):  # pragma: no cover - simple awaitable
        return self.skills[name]


def test_divergent_convergent_solver():
    graph = {
        "A": ["B", "C"],
        "B": ["D"],
        "C": ["E"],
        "E": ["D"],
        "D": [],
    }
    solver = DivergentConvergentSolver(MultiHopAssociator(graph), ReflectionModule())
    path, score = solver.solve("A", "D", ["B", "C"])
    assert path == ["A", "C", "E", "D"]
    assert score == 4


def test_planner_solve_delegates_solver():
    graph = {
        "A": ["B", "C"],
        "B": ["D"],
        "C": ["E"],
        "E": ["D"],
        "D": [],
    }
    solver = DivergentConvergentSolver(MultiHopAssociator(graph), ReflectionModule())
    planner = Planner(solver=solver)
    result = planner.solve({"start": "A", "goal": "D", "strategies": ["B", "C"]})
    assert result == ["A", "C", "E", "D"]


def test_executor_selects_best_plan():
    lib = DummySkillLibrary()
    executor = Executor(lib)
    plans = [("plan_a", 0.1), ("plan_b", 0.9)]
    result = executor.execute(plans)
    assert result["plan_b"] == "B"
