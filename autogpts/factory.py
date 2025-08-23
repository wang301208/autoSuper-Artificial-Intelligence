"""Public API for spawning agents from blueprint files."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from autogpt.config import Config
from autogpt.core.resource.model_providers import ChatModelProvider
from autogpt.file_storage.base import FileStorage

from agent_factory import create_agent_from_blueprint


def spawn_agent(
    blueprint_path: str | Path,
    *,
    config: Config,
    llm_provider: ChatModelProvider,
    file_storage: FileStorage,
):
    """Create a new AutoGPT agent from a blueprint.

    Parameters
    ----------
    blueprint_path: str | Path
        Path to the blueprint YAML file describing the agent.
    config: Config
        Application configuration to apply to the agent.
    llm_provider: ChatModelProvider
        LLM provider used for the agent's thinking.
    file_storage: FileStorage
        Storage backend for the agent's file operations.
    """
    return create_agent_from_blueprint(
        blueprint_path,
        config=config,
        llm_provider=llm_provider,
        file_storage=file_storage,
    )


__all__ = ["spawn_agent"]
