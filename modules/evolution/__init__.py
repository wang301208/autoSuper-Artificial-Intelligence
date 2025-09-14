"""Evolution package containing specialized agent implementations."""

from abc import ABC, abstractmethod

from .replay_buffer import ReplayBuffer
from .evolving_cognitive_architecture import (
    EvolvingCognitiveArchitecture,
    GeneticAlgorithm as EvolutionGeneticAlgorithm,
)
from .self_evolving_cognition import SelfEvolvingCognition
from .self_evolving_ai_architecture import SelfEvolvingAIArchitecture
from .evolution_engine import EvolutionEngine
from .adapter import EvolutionModule

try:  # optional dependencies
    from .ppo import PPO, PPOConfig
    from .a3c import A3C, A3CConfig
    from .sac import SAC, SACConfig
except Exception:  # pragma: no cover - algorithms require torch
    PPO = PPOConfig = A3C = A3CConfig = SAC = SACConfig = None  # type: ignore

try:  # Attempt to leverage AutoGPT's agent base when available
    from autogpts.autogpt.autogpt.agents.base import BaseAgent as AutoGPTBaseAgent
except Exception:  # pragma: no cover - fallback when dependencies missing
    AutoGPTBaseAgent = None  # type: ignore


if AutoGPTBaseAgent is None:
    class Agent(ABC):  # type: ignore[too-many-ancestors]
        """Abstract base agent for the evolution package.

        Subclasses should implement :meth:`perform`, which executes the agent's
        primary behaviour.
        """

        @abstractmethod
        def perform(self, *args, **kwargs):
            """Execute the agent's primary behaviour."""
            raise NotImplementedError


else:
    class Agent(AutoGPTBaseAgent, ABC):
        """Abstract base agent for the evolution package.

        Subclasses should implement :meth:`perform`, which executes the agent's
        primary behaviour. When AutoGPT's BaseAgent is available the class
        inherits from it, enabling integration with the broader AutoGPT
        ecosystem.
        """

        @abstractmethod
        def perform(self, *args, **kwargs):
            """Execute the agent's primary behaviour."""
            raise NotImplementedError


__all__ = [
    "Agent",
    "ReplayBuffer",
    "EvolvingCognitiveArchitecture",
    "EvolutionGeneticAlgorithm",
    "SelfEvolvingCognition",
    "SelfEvolvingAIArchitecture",
    "EvolutionEngine",
    "PPO",
    "PPOConfig",
    "A3C",
    "A3CConfig",
    "SAC",
    "SACConfig",
    "EvolutionModule",
]
