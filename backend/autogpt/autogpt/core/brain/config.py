from __future__ import annotations

from enum import Enum

from pydantic import Field

from autogpt.core.configuration import SystemConfiguration, UserConfigurable

from modules.brain.state import BrainRuntimeConfig


class BrainBackend(str, Enum):
    """Selects which cognitive backend drives ``BaseAgent.propose_action``."""

    LLM = "llm"
    TRANSFORMER = "transformer"
    WHOLE_BRAIN = "whole_brain"


class TransformerBrainConfig(SystemConfiguration):
    """Configuration options for the internal transformer brain."""

    layers: int = UserConfigurable(default=2)
    """Number of transformer encoder layers."""

    heads: int = UserConfigurable(default=4)
    """Number of attention heads in each transformer layer."""

    dim: int = UserConfigurable(default=256)
    """Dimensionality of the model."""

    dropout: float = UserConfigurable(default=0.1)
    """Dropout probability used in transformer layers."""

    weights_path: str | None = UserConfigurable(default=None)
    """Optional path to load pretrained weights at initialization."""

    dataset_logging_path: str | None = UserConfigurable(default=None)
    """If set, write per-cycle brain samples to this JSONL file."""

    # Training configuration
    learning_rate: float = UserConfigurable(default=1e-3)
    """Learning rate used during training."""

    epochs: int = UserConfigurable(default=10)
    """Number of training epochs."""

    batch_size: int = UserConfigurable(default=4)
    """Batch size for the training dataloader."""


class WholeBrainRuntimeSettings(SystemConfiguration):
    """Toggles mirrored onto :class:`modules.brain.state.BrainRuntimeConfig`."""

    use_neuromorphic: bool = UserConfigurable(default=True, from_env="WHOLE_BRAIN_USE_NEUROMORPHIC")
    enable_multi_dim_emotion: bool = UserConfigurable(
        default=True, from_env="WHOLE_BRAIN_ENABLE_MULTI_DIM_EMOTION"
    )
    enable_emotion_decay: bool = UserConfigurable(
        default=True, from_env="WHOLE_BRAIN_ENABLE_EMOTION_DECAY"
    )
    enable_curiosity_feedback: bool = UserConfigurable(
        default=True, from_env="WHOLE_BRAIN_ENABLE_CURIOSITY_FEEDBACK"
    )
    enable_self_learning: bool = UserConfigurable(
        default=True, from_env="WHOLE_BRAIN_ENABLE_SELF_LEARNING"
    )
    enable_personality_modulation: bool = UserConfigurable(
        default=True, from_env="WHOLE_BRAIN_ENABLE_PERSONALITY_MODULATION"
    )
    enable_plan_logging: bool = UserConfigurable(
        default=True, from_env="WHOLE_BRAIN_ENABLE_PLAN_LOGGING"
    )
    metrics_enabled: bool = UserConfigurable(
        default=True, from_env="WHOLE_BRAIN_METRICS_ENABLED"
    )

    def to_runtime(self) -> BrainRuntimeConfig:
        """Create a :class:`BrainRuntimeConfig` dataclass reflecting the settings."""

        return BrainRuntimeConfig(
            use_neuromorphic=self.use_neuromorphic,
            enable_multi_dim_emotion=self.enable_multi_dim_emotion,
            enable_emotion_decay=self.enable_emotion_decay,
            enable_curiosity_feedback=self.enable_curiosity_feedback,
            enable_self_learning=self.enable_self_learning,
            enable_personality_modulation=self.enable_personality_modulation,
            enable_plan_logging=self.enable_plan_logging,
            metrics_enabled=self.metrics_enabled,
        )


class WholeBrainConfig(SystemConfiguration):
    """Configuration wrapper for :class:`modules.brain.whole_brain.WholeBrainSimulation`."""

    neuromorphic_encoding: str = UserConfigurable(
        default="rate", from_env="WHOLE_BRAIN_ENCODING"
    )
    encoding_steps: int = UserConfigurable(
        default=5, from_env="WHOLE_BRAIN_ENCODING_STEPS"
    )
    encoding_time_scale: float = UserConfigurable(
        default=1.0, from_env="WHOLE_BRAIN_ENCODING_TIME_SCALE"
    )
    max_neurons: int = UserConfigurable(default=128, from_env="WHOLE_BRAIN_MAX_NEURONS")
    max_cache_size: int = UserConfigurable(
        default=8, from_env="WHOLE_BRAIN_MAX_CACHE_SIZE"
    )
    runtime: WholeBrainRuntimeSettings = Field(default_factory=WholeBrainRuntimeSettings)

    def to_simulation_kwargs(self) -> dict:
        """Return keyword arguments used to initialize the simulation."""

        runtime = self.runtime.to_runtime()
        return {
            "config": runtime,
            "neuromorphic": runtime.use_neuromorphic,
            "neuromorphic_encoding": self.neuromorphic_encoding,
            "encoding_steps": self.encoding_steps,
            "encoding_time_scale": self.encoding_time_scale,
            "max_neurons": self.max_neurons,
            "max_cache_size": self.max_cache_size,
        }


__all__ = [
    "BrainBackend",
    "TransformerBrainConfig",
    "WholeBrainConfig",
    "WholeBrainRuntimeSettings",
]
