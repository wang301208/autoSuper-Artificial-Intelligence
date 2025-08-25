"""Utilities for constructing agents from blueprint files.

This module parses agent blueprint YAML files and instantiates fully
configured :class:`autogpt.agents.agent.Agent` objects. Blueprints
specify the core prompt for the agent as well as which tools it is
permitted to use. The factory uses AutoGPT's configuration and capability
system to ensure the created agent only has access to the specified
tools.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import ast
import logging
from json import JSONDecodeError

import yaml
from jsonschema import validate

from autogpt.agents.agent import Agent
from autogpt.agent_factory.configurators import create_agent
from autogpt.command_decorator import AUTO_GPT_COMMAND_IDENTIFIER
from autogpt.config import AIProfile, Config
from autogpt.config.ai_directives import AIDirectives
from autogpt.core.resource.model_providers import ChatModelProvider
from autogpt.file_storage.base import FileStorage
from autogpt.models.command_registry import CommandRegistry

from autogpts.autogpt.autogpt.core.errors import AutoGPTError

from capability.librarian import Librarian
from org_charter import io as charter_io
from common.security import SkillSecurityError, SAFE_BUILTINS, _verify_skill


logger = logging.getLogger(__name__)


def _parse_blueprint(path: Path) -> dict:
    """Load and validate a blueprint file.

    Parameters
    ----------
    path: Path
        Path to a YAML file conforming to ``schemas/agent_blueprint.yaml``.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    validate(instance=data, schema=charter_io.BLUEPRINT_SCHEMA)
    return data


def _load_additional_tool(
    name: str, registry: CommandRegistry, repo_path: Path
) -> None:
    """Load a tool from the skill library and register it.

    This uses the :mod:`capability` package which serves as a very light-weight
    plugin system.  Skills are stored as Python source files and are expected to
    expose callables decorated with :func:`autogpt.command_decorator.command`.
    """
    librarian = Librarian(str(repo_path))
    try:
        code, meta = librarian.get_skill(name)
    except (OSError, PermissionError, JSONDecodeError) as err:
        logger.exception("Failed to retrieve tool '%s' from skill library", name)
        raise AutoGPTError(f"Failed to load tool '{name}'") from err

    try:
        _verify_skill(name, code, meta)
    except SkillSecurityError as err:
        logger.warning("Rejected tool '%s': %s", name, err.cause)
        raise

    namespace: dict[str, object] = {}
    try:
        parsed = ast.parse(code, mode="exec")
        exec(compile(parsed, filename=name, mode="exec"), {"__builtins__": SAFE_BUILTINS}, namespace)
    except (SyntaxError, TypeError, ValueError) as err:
        logger.exception("Error executing tool '%s'", name)
        raise AutoGPTError(f"Failed to initialize tool '{name}'") from err

    for obj in namespace.values():
        if getattr(obj, AUTO_GPT_COMMAND_IDENTIFIER, False):
            cmd = getattr(obj, "command", None)
            if cmd:
                registry.register(cmd)
                break


def _filter_authorized_tools(registry: CommandRegistry, allowed: Iterable[str]) -> None:
    allowed = set(allowed)
    for name, cmd in list(registry.commands.items()):
        if name not in allowed:
            registry.unregister(cmd)


def create_agent_from_blueprint(
    blueprint_path: Path | str,
    config: Config,
    llm_provider: ChatModelProvider,
    file_storage: FileStorage,
) -> Agent:
    """Instantiate an :class:`Agent` from a blueprint file.

    Parameters
    ----------
    blueprint_path:
        Path to a YAML blueprint file describing the agent.
    config:
        Application configuration to use for the new agent.
    llm_provider:
        Model provider used for prompting.
    file_storage:
        File storage backend for the agent.
    """
    path = Path(blueprint_path)
    blueprint = _parse_blueprint(path)

    profile = AIProfile(
        ai_name=blueprint["role_name"],
        ai_role=blueprint["role_name"],
        ai_goals=[blueprint["core_prompt"]],
    )

    directives = AIDirectives.from_file(config.prompt_settings_file)

    agent = create_agent(
        agent_id=blueprint["role_name"],
        task=blueprint["core_prompt"],
        ai_profile=profile,
        directives=directives,
        app_config=config,
        file_storage=file_storage,
        llm_provider=llm_provider,
    )

    # Restrict commands to authorised tools
    _filter_authorized_tools(
        agent.command_registry, blueprint.get("authorized_tools", [])
    )

    # Attempt to load additional tools from the capability library if they were
    # requested but not part of the default command set.
    repo_root = Path.cwd()
    for tool_name in blueprint.get("authorized_tools", []):
        if tool_name not in agent.command_registry.commands:
            _load_additional_tool(tool_name, agent.command_registry, repo_root)

    # Store subscribed topics for potential later use by messaging layers.
    setattr(agent, "subscribed_topics", blueprint.get("subscribed_topics", []))
    return agent


__all__ = ["create_agent_from_blueprint"]
