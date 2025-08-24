from __future__ import annotations

from autogpt.core.configuration.schema import (
    SystemConfiguration,
    SystemSettings,
    UserConfigurable,
)


class LearningConfiguration(SystemConfiguration):
    """Configuration options for experience-based learning."""

    enabled: bool = UserConfigurable(
        default=False, description="Enable learning from stored experiences"
    )
    learning_rate: float = UserConfigurable(
        default=0.001, description="Learning rate for model updates"
    )
    batch_size: int = UserConfigurable(
        default=32, description="Batch size of experiences used per update"
    )


class LearningSettings(SystemSettings):
    """Settings wrapper for the ExperienceLearner."""

    configuration: LearningConfiguration
