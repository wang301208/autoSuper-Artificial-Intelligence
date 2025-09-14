import asyncio
from datetime import datetime
from uuid import uuid4

import pytest
from forge.sdk.model import Task

from autogpt.agent_factory.configurators import (
    configure_agent_with_state,
    create_agent_state,
)
from autogpt.agent_manager import AgentManager
from autogpt.config.ai_directives import AIDirectives
from autogpt.config.ai_profile import AIProfile


@pytest.mark.asyncio
async def test_multiple_agents_shared_workspace(config, storage, llm_provider):
    directives = AIDirectives.from_file(config.prompt_settings_file)
    ai_profile = AIProfile(ai_name="Tester", ai_role="testing", ai_goals=[])

    def make_task():
        now = datetime.now()
        return Task(
            input="Test task",
            additional_input=None,
            created_at=now,
            modified_at=now,
            task_id=str(uuid4()),
            artifacts=[],
        )

    state1 = create_agent_state("agent1", make_task(), ai_profile, directives, config)
    state1.workspace_id = "shared"
    agent1 = configure_agent_with_state(state1, config, storage, llm_provider)

    state2 = create_agent_state("agent2", make_task(), ai_profile, directives, config)
    state2.workspace_id = "shared"
    agent2 = configure_agent_with_state(state2, config, storage, llm_provider)

    await asyncio.gather(
        agent1.workspace.write_file("file1.txt", "agent1"),
        agent2.workspace.write_file("file2.txt", "agent2"),
    )

    await asyncio.gather(agent1.save_state(), agent2.save_state())

    manager = AgentManager(storage)
    restored1 = configure_agent_with_state(
        manager.load_agent_state("agent1"), config, storage, llm_provider
    )
    restored2 = configure_agent_with_state(
        manager.load_agent_state("agent2"), config, storage, llm_provider
    )

    assert restored1.workspace.read_file("file1.txt") == "agent1"
    assert restored1.workspace.read_file("file2.txt") == "agent2"
    assert restored2.workspace.read_file("file1.txt") == "agent1"
    assert restored2.workspace.read_file("file2.txt") == "agent2"
