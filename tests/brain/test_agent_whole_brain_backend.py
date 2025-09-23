import asyncio
import importlib.util
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend/autogpt"))
sys.path.insert(0, str(ROOT / "modules"))

# Provide a lightweight stub for the plugin template dependency
import types

auto_plugin = types.SimpleNamespace(AutoGPTPluginTemplate=type("Plugin", (), {}))
sys.modules.setdefault("auto_gpt_plugin_template", auto_plugin)
sentry_stub = types.SimpleNamespace(capture_exception=lambda *args, **kwargs: None)
sys.modules.setdefault("sentry_sdk", sentry_stub)

if importlib.util.find_spec("pydantic") is None:  # pragma: no cover - optional dependency absent
    pytestmark = pytest.mark.skip(reason="pydantic not available for whole-brain integration test")
    Agent = AgentConfiguration = AgentSettings = None  # type: ignore
    BrainBackend = None  # type: ignore
    Action = ActionSuccessResult = Episode = None  # type: ignore
else:
    try:  # pragma: no cover - gracefully handle missing heavy dependencies
        from autogpt.agents.agent import Agent, AgentConfiguration, AgentSettings
    except ModuleNotFoundError:  # e.g. dependency resolution failed at runtime
        Agent = AgentConfiguration = AgentSettings = None  # type: ignore
    from autogpt.core.brain.config import BrainBackend
    from autogpt.models.action_history import Action, ActionSuccessResult, Episode


def test_agent_with_whole_brain_backend_proposes_internal_action():
    if Agent is None:
        pytest.skip("Agent dependencies not available in this test environment")

    async def _run_test() -> None:
        llm_provider = Mock()
        llm_provider.create_chat_completion = AsyncMock()
        llm_provider.count_tokens = Mock(return_value=0)

        command_registry = Mock()
        command_registry.list_available_commands.return_value = []
        command_registry.get_command.return_value = None

        file_storage = Mock()
        legacy_config = SimpleNamespace(
            event_bus_backend="inmemory",
            event_bus_redis_host="localhost",
            event_bus_redis_port=0,
            event_bus_redis_password="",
        )

        config = AgentConfiguration(brain_backend=BrainBackend.WHOLE_BRAIN, big_brain=True)
        config.whole_brain.runtime.enable_self_learning = False
        config.whole_brain.runtime.metrics_enabled = True
        settings = AgentSettings(config=config)

        agent = Agent(
            settings=settings,
            llm_provider=llm_provider,
            command_registry=command_registry,
            file_storage=file_storage,
            legacy_config=legacy_config,
        )

        agent.event_history.episodes.extend(
            [
                Episode(
                    action=Action(name="bootstrap", args={}, reasoning="initialise"),
                    result=ActionSuccessResult(outputs="ok"),
                    summary="Boot sequence completed.",
                ),
                Episode(
                    action=Action(name="observe", args={}, reasoning="check status"),
                    result=ActionSuccessResult(outputs="environment stable"),
                    summary="Status is nominal.",
                ),
            ]
        )

        command, args, thoughts = await agent.propose_action()

        assert command == "internal_brain_action"
        assert args["intention"]
        assert thoughts["backend"] == "whole_brain"
        assert isinstance(thoughts["plan"], list)
        assert thoughts["metrics"]  # telemetry forwarded from WholeBrainSimulation
        llm_provider.create_chat_completion.assert_not_called()

    asyncio.run(_run_test())
