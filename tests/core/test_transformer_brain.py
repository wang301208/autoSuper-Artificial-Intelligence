import json
import sys
import types
from pathlib import Path

import pytest
import torch
from unittest.mock import AsyncMock, Mock, patch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend/autogpt"))
sys.path.insert(0, str(ROOT / "modules"))

# Provide minimal stubs for external dependencies
auto_plugin = types.SimpleNamespace(AutoGPTPluginTemplate=type("Plugin", (), {}))
sys.modules.setdefault("auto_gpt_plugin_template", auto_plugin)

try:  # Attempt to import the full agent stack
    from autogpt.agents.base import BaseAgent, BaseAgentSettings, BaseAgentConfiguration
    from autogpt.models.action_history import ActionSuccessResult
except Exception:  # pragma: no cover - environment limitations
    BaseAgent = BaseAgentSettings = BaseAgentConfiguration = None
    ActionSuccessResult = type("ActionSuccessResult", (), {"__init__": lambda self, outputs: None})

from autogpt.core.brain.config import TransformerBrainConfig
from autogpt.core.brain.encoding import build_brain_inputs
from autogpt.core.brain.transformer_brain import TransformerBrain
from autogpt.core.brain.train_transformer_brain import ObservationActionDataset
from autogpt.models.action_history import Action, Episode


if BaseAgent is not None:
    class DummyAgent(BaseAgent):  # pragma: no cover - skipped if BaseAgent is None
        async def execute(self, command_name: str, command_args: dict[str, str] = {}, user_input: str = ""):
            return ActionSuccessResult(outputs="done")

        def parse_and_process_response(self, llm_response, prompt, scratchpad):
            return "noop", {}, {}


def test_transformer_brain_outputs_thought_and_action():
    config = TransformerBrainConfig()
    brain = TransformerBrain(config)

    workspace = Mock()
    memory = Mock()
    workspace.get_observation.return_value = torch.zeros(config.dim)
    memory.get_context.return_value = torch.zeros(config.dim)

    thought = brain.think(workspace.get_observation(), memory.get_context())
    command, args, info = brain.propose_action(thought)

    assert thought.shape[-1] == config.dim
    assert isinstance(info["action"], list)
    assert command == "internal_brain_action"


def test_build_brain_inputs_produces_dense_features():
    episodes = [
        Episode(
            action=Action(name="do_task", args={"foo": "bar"}, reasoning="evaluate options"),
            result=ActionSuccessResult(outputs="ok"),
            summary="Executed do_task successfully.",
        )
    ]
    agent = types.SimpleNamespace(
        config=types.SimpleNamespace(cycle_count=3, cycle_budget=10),
        event_history=types.SimpleNamespace(episodes=episodes),
        state=types.SimpleNamespace(task=types.SimpleNamespace(input="ship feature", additional_input="")),
        ai_profile=types.SimpleNamespace(ai_goals=["launch product"]),
        directives=types.SimpleNamespace(general_guidelines=["stay helpful"]),
    )

    observation, memory_ctx = build_brain_inputs(agent, dim=16)

    assert observation.shape == (16,)
    assert memory_ctx.shape == (16,)
    assert torch.count_nonzero(observation) > 0
    assert torch.count_nonzero(memory_ctx) > 0


def test_observation_action_dataset_from_jsonl(tmp_path):
    sample_path = tmp_path / "brain_samples.jsonl"
    sample = {
        "observation": [0.1, 0.2, 0.3],
        "memory": [0.4, 0.5, 0.6],
        "action_index": 2,
    }
    sample_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    dataset = ObservationActionDataset.from_jsonl(sample_path, dim=4)
    obs, mem, action = dataset[0]

    assert obs.shape == (4,)
    assert mem.shape == (4,)
    assert action.item() == 2


@pytest.mark.asyncio
@pytest.mark.skipif(BaseAgent is None, reason="BaseAgent dependencies not available")
async def test_agent_uses_brain_when_enabled():
    brain_instance = Mock(spec=TransformerBrain)
    brain_instance.think.return_value = torch.zeros(256)
    brain_instance.propose_action.return_value = ("internal_brain_action", {}, {})
    with patch("autogpt.agents.base.TransformerBrain", return_value=brain_instance):
        llm_provider = Mock()
        llm_provider.create_chat_completion = AsyncMock()
        prompt_strategy = Mock()
        command_registry = Mock()
        file_storage = Mock()
        legacy_config = types.SimpleNamespace(
            event_bus_backend="inmemory",
            event_bus_redis_host="localhost",
            event_bus_redis_port=0,
            event_bus_redis_password="",
        )

        settings = BaseAgentSettings(
            config=BaseAgentConfiguration(big_brain=True),
        )

        agent = DummyAgent(
            settings=settings,
            llm_provider=llm_provider,
            prompt_strategy=prompt_strategy,
            command_registry=command_registry,
            file_storage=file_storage,
            legacy_config=legacy_config,
        )

        await agent.propose_action()

    brain_instance.think.assert_called_once()
    brain_instance.propose_action.assert_called_once()
    llm_provider.create_chat_completion.assert_not_called()
