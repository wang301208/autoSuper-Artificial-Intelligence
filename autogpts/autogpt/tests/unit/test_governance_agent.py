import importlib.util
import types
from pathlib import Path
import sys
from typing import Optional

import pytest

# Stub out heavy dependencies before importing the governance module
agent_pkg = types.ModuleType("autogpt.core.agent")
layered_pkg = types.ModuleType("autogpt.core.agent.layered")


class _DummyLayeredAgent:
    def __init__(self, *args, next_layer=None, **kwargs):
        self.next_layer = next_layer

    def route_task(self, task, *args, **kwargs):
        if self.next_layer is not None:
            return self.next_layer.route_task(task, *args, **kwargs)
        raise NotImplementedError


layered_pkg.LayeredAgent = _DummyLayeredAgent
agent_pkg.LayeredAgent = _DummyLayeredAgent
agent_pkg.Agent = object
agent_pkg.AgentSettings = object
agent_pkg.SimpleAgent = object
sys.modules["autogpt.core.agent"] = agent_pkg
sys.modules["autogpt.core.agent.layered"] = layered_pkg

# Import governance module directly to avoid heavy package dependencies
_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "autogpt"
    / "agents"
    / "layers"
    / "governance.py"
)
spec = importlib.util.spec_from_file_location("governance", _MODULE_PATH)
governance = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(governance)  # type: ignore[call-arg]
governance.GovernancePolicy.update_forward_refs(
    Path=Path, Charter=governance.Charter, Optional=Optional
)
governance.GovernanceAgentSettings.update_forward_refs(
    GovernancePolicy=governance.GovernancePolicy
)

GovernanceAgent = governance.GovernanceAgent
GovernanceAgentSettings = governance.GovernanceAgentSettings
GovernancePolicy = governance.GovernancePolicy


def make_agent(role: str = "assistant") -> GovernanceAgent:
    settings = GovernanceAgentSettings(
        name="gov",
        description="gov",
        policy=GovernancePolicy(
            charter_name="example",
            role=role,
            charter_path=Path(__file__).resolve().parents[2] / "data" / "charter",
        ),
    )
    return GovernanceAgent(settings=settings)


def test_loads_charter() -> None:
    agent = make_agent()
    assert agent.charter.name == "Example Charter"


def test_allows_task_for_role() -> None:
    agent = make_agent()
    result = agent.route_task("answer questions")
    assert result == "answer questions"


def test_blocks_disallowed_task() -> None:
    agent = make_agent()
    with pytest.raises(PermissionError) as exc:
        agent.route_task("forbidden")
    assert "not permitted" in str(exc.value)


def test_blocks_unknown_role() -> None:
    agent = make_agent(role="stranger")
    with pytest.raises(PermissionError) as exc:
        agent.route_task("answer questions")
    assert "not defined" in str(exc.value)
