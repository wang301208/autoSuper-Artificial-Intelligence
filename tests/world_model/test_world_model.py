"""Tests for the world model module."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _load(name: str, path: str):
    abs_path = Path(__file__).resolve().parents[2] / path
    spec = importlib.util.spec_from_file_location(name, str(abs_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Create a package-like structure for ``backend`` so modules can be imported
backend_pkg = types.ModuleType("backend")
backend_pkg.__path__ = []
sys.modules["backend"] = backend_pkg

wm_module = _load("backend.world_model", "backend/world_model/__init__.py")
backend_pkg.world_model = wm_module
WorldModel = wm_module.WorldModel


def test_state_updates_and_predictions():
    wm = WorldModel()

    # Add a task and update resources twice to exercise the learning component
    wm.add_task("task1", {"description": "test"})
    wm.update_resources("agent1", {"cpu": 1.0, "memory": 1.0})
    wm.update_resources("agent1", {"cpu": 3.0, "memory": 3.0})
    wm.record_action("agent1", "run")

    state = wm.get_state()
    assert "task1" in state["tasks"]
    assert state["resources"]["agent1"]["cpu"] == 3.0
    assert state["actions"][0]["action"] == "run"

    # The prediction should be the moving average of the two updates
    pred_agent = wm.predict("agent1")
    assert pred_agent["cpu"] == 2.0
    assert pred_agent["memory"] == 2.0

    pred_all = wm.predict()
    assert pred_all["avg_cpu"] == 2.0
    assert pred_all["avg_memory"] == 2.0

