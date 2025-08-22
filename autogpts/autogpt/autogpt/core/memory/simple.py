import json
import logging

from autogpt.core.configuration import Configurable, SystemConfiguration, SystemSettings
from autogpt.core.memory.base import Memory
from autogpt.core.workspace import Workspace


class MemoryConfiguration(SystemConfiguration):
    pass


class MemorySettings(SystemSettings):
    configuration: MemoryConfiguration


class MessageHistory:
    def __init__(self, previous_message_history: list[str]):
        self._message_history = previous_message_history

    def append(self, message: str) -> None:
        self._message_history.append(message)

    def as_list(self) -> list[str]:
        return self._message_history


class SimpleMemory(Memory, Configurable):
    default_settings = MemorySettings(
        name="simple_memory",
        description="A simple memory.",
        configuration=MemoryConfiguration(),
    )

    def __init__(
        self,
        settings: MemorySettings,
        logger: logging.Logger,
        workspace: Workspace,
    ):
        self._configuration = settings.configuration
        self._logger = logger
        self._workspace = workspace
        self._message_history = self._load_message_history(workspace)

    @staticmethod
    def _load_message_history(workspace: Workspace):
        message_history_path = workspace.get_path("message_history.json")
        if message_history_path.exists():
            with message_history_path.open("r") as f:
                message_history = json.load(f)
        else:
            message_history = []
        return MessageHistory(message_history)

    def add(self, message: str) -> None:
        """Store a message in persistent memory."""
        self._message_history.append(message)
        path = self._workspace.get_path("message_history.json")
        with path.open("w") as f:
            json.dump(self._message_history.as_list(), f)

    def get(self, limit: int | None = None) -> list[str]:
        """Return messages from memory.

        Args:
            limit: If provided, return only the most recent `limit` messages.

        Returns:
            List of stored messages.
        """
        messages = self._message_history.as_list()
        return messages[-limit:] if limit else messages
