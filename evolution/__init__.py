"""Evolution package containing specialized agent implementations."""

from abc import ABC, abstractmethod

try:  # Attempt to leverage AutoGPT's agent base when available
    from autogpts.autogpt.autogpt.agents.base import BaseAgent as AutoGPTBaseAgent
except Exception:  # pragma: no cover - fallback when dependencies missing
    AutoGPTBaseAgent = ABC  # type: ignore


class Agent(AutoGPTBaseAgent, ABC):
    """Abstract base agent for the evolution package.

    Subclasses should implement :meth:`perform`, which executes the agent's
    primary behaviour. When AutoGPT's BaseAgent is available the class inherits
    from it, enabling integration with the broader AutoGPT ecosystem.
    """

    @abstractmethod
    def perform(self, *args, **kwargs):
        """Execute the agent's primary behaviour."""
        raise NotImplementedError
