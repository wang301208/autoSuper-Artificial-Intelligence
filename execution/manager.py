"""Agent lifecycle manager.

This component watches for blueprint changes and ensures that running agents are
reloaded or spawned to reflect the latest blueprints. Reload results are
published on the global event bus for observability.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from agent_factory import create_agent_from_blueprint
from events import EventBus
from org_charter.watchdog import BlueprintWatcher
from autogpt.config import Config
from autogpt.core.resource.model_providers import ChatModelProvider
from autogpt.file_storage.base import FileStorage
from autogpt.agents.agent import Agent


class AgentLifecycleManager:
    """Manage running agents and reload them when blueprints change."""

    def __init__(
        self,
        config: Config,
        llm_provider: ChatModelProvider,
        file_storage: FileStorage,
        event_bus: EventBus,
    ) -> None:
        self._config = config
        self._llm_provider = llm_provider
        self._file_storage = file_storage
        self._event_bus = event_bus
        self._agents: Dict[str, Agent] = {}
        self._watcher = BlueprintWatcher(self._on_blueprint_change)
        self._watcher.start()

    # ------------------------------------------------------------------
    # Blueprint change handling
    # ------------------------------------------------------------------
    def _on_blueprint_change(self, path: Path) -> None:
        name = path.stem.split("_v")[0]
        try:
            agent = create_agent_from_blueprint(
                path, self._config, self._llm_provider, self._file_storage
            )
            previous = self._agents.get(name)
            if previous is not None:
                _shutdown_agent(previous)
                action = "restarted"
            else:
                action = "spawned"
            self._agents[name] = agent
            self._event_bus.publish(
                "agent.lifecycle",
                {"agent": name, "action": action, "path": str(path)},
            )
        except Exception as exc:  # pragma: no cover - logging path
            self._event_bus.publish(
                "agent.lifecycle",
                {
                    "agent": name,
                    "action": "failed",
                    "path": str(path),
                    "error": str(exc),
                },
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Stop watching and shut down all managed agents."""
        self._watcher.stop()
        for agent in self._agents.values():
            _shutdown_agent(agent)
        self._agents.clear()


def _shutdown_agent(agent: Agent) -> None:
    """Attempt to gracefully shut down an agent if it supports it."""
    for method_name in ("shutdown", "stop", "close"):
        method = getattr(agent, method_name, None)
        if callable(method):
            try:
                method()  # type: ignore[misc]
            except Exception:  # pragma: no cover - best effort
                pass
            break


__all__ = ["AgentLifecycleManager"]
