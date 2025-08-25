"""Agent lifecycle manager.

This component watches for blueprint changes and ensures that running agents are
reloaded or spawned to reflect the latest blueprints. Reload results are
published on the global event bus for observability.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Dict

import psutil

from agent_factory import create_agent_from_blueprint
from events import EventBus
from monitoring import SystemMetricsCollector
from org_charter.watchdog import BlueprintWatcher
from autogpt.config import Config
from autogpt.core.resource.model_providers import ChatModelProvider
from autogpt.file_storage.base import FileStorage
from autogpt.agents.agent import Agent
from .scheduler import Scheduler


class AgentLifecycleManager:
    """Manage running agents and reload them when blueprints change."""

    def __init__(
        self,
        config: Config,
        llm_provider: ChatModelProvider,
        file_storage: FileStorage,
        event_bus: EventBus,
        scheduler: Scheduler | None = None,
    ) -> None:
        self._config = config
        self._llm_provider = llm_provider
        self._file_storage = file_storage
        self._event_bus = event_bus
        self._scheduler = scheduler
        self._agents: Dict[str, Agent] = {}
        self._resources: Dict[str, Dict[str, float]] = {}
        self._metrics = SystemMetricsCollector(event_bus)
        self._metrics.start()
        self._event_bus.subscribe("agent.resource", self._on_resource_event)
        self._resource_stop = threading.Event()
        self._resource_thread = threading.Thread(
            target=self._resource_manager, daemon=True
        )
        self._resource_thread.start()
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
                self._metrics.unregister(name)
                action = "restarted"
            else:
                action = "spawned"
            self._agents[name] = agent
            self._metrics.register(name, getattr(agent, "pid", os.getpid()))
            self._resources[name] = {
                "cpu": 0.0,
                "memory": 0.0,
                "last_active": time.time(),
            }
            if self._scheduler:
                self._scheduler.add_agent(name)
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
        self._resource_stop.set()
        self._resource_thread.join()
        self._metrics.stop()
        for name, agent in list(self._agents.items()):
            _shutdown_agent(agent)
            if self._scheduler:
                self._scheduler.remove_agent(name)
        self._agents.clear()

    # ------------------------------------------------------------------
    def _on_resource_event(self, event: Dict[str, float]) -> None:
        name = event.get("agent")
        if name not in self._resources:
            return
        data = self._resources[name]
        data["cpu"] = event.get("cpu", 0.0)
        data["memory"] = event.get("memory", 0.0)
        if self._scheduler:
            self._scheduler.update_agent(name, data["cpu"], data["memory"])
        if data["cpu"] > 1.0:
            data["last_active"] = time.time()

    def _resource_manager(self) -> None:
        idle_timeout = 30.0
        pressure_threshold = 80.0
        while not self._resource_stop.wait(5.0):
            now = time.time()
            for name, data in list(self._resources.items()):
                if now - data.get("last_active", now) > idle_timeout:
                    agent = self._agents.get(name)
                    if agent is not None:
                        _shutdown_agent(agent)
                        self._metrics.unregister(name)
                        del self._agents[name]
                        del self._resources[name]
                        if self._scheduler:
                            self._scheduler.remove_agent(name)
                        self._event_bus.publish(
                            "agent.lifecycle", {"agent": name, "action": "reclaimed"}
                        )
            cpu_total = psutil.cpu_percent()
            mem_total = psutil.virtual_memory().percent
            if cpu_total > pressure_threshold or mem_total > pressure_threshold:
                if self._resources:
                    heavy = max(
                        self._resources.items(), key=lambda item: item[1].get("cpu", 0.0)
                    )[0]
                    self._event_bus.publish(
                        "agent.resource",
                        {"agent": heavy, "action": "throttle"},
                    )


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
