"""Core exception hierarchy for AutoGPT."""
from __future__ import annotations


class AutoGPTError(Exception):
    """Base class for all AutoGPT specific exceptions."""


class ConfigurationError(AutoGPTError):
    """Raised for configuration related issues."""


class PluginError(AutoGPTError):
    """Raised when a plugin operation fails."""


class EventBusError(AutoGPTError):
    """Raised for problems related to the event bus."""
