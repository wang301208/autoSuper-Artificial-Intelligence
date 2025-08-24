"""The planning system organizes the Agent's activities."""

from autogpt.core.planning.schema import Task, TaskStatus, TaskType

__all__ = [
    "PlannerSettings",
    "SimplePlanner",
    "ReasoningPlanner",
    "PlanResult",
    "Task",
    "TaskStatus",
    "TaskType",
]


def __getattr__(name: str):
    if name in {"PlannerSettings", "SimplePlanner"}:
        from autogpt.core.planning.simple import PlannerSettings, SimplePlanner

        return {"PlannerSettings": PlannerSettings, "SimplePlanner": SimplePlanner}[name]
    if name in {"ReasoningPlanner", "PlanResult"}:
        from autogpt.core.planning.reasoner import PlanResult, ReasoningPlanner

        return {"ReasoningPlanner": ReasoningPlanner, "PlanResult": PlanResult}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
