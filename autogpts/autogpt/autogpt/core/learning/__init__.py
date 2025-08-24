"""Learning utilities for adapting the agent from past experiences."""
from __future__ import annotations

import logging
from typing import Iterable

from autogpt.core.configuration.learning import LearningConfiguration


class ExperienceLearner:
    """Learn from stored experiences to update model parameters."""

    def __init__(
        self,
        memory: Iterable,
        config: LearningConfiguration,
        logger: logging.Logger | None = None,
    ) -> None:
        self._memory = memory
        self._config = config
        self._logger = logger or logging.getLogger(__name__)

    def learn_from_experience(self) -> None:
        """Read past interactions from memory and update the model."""
        if not self._config.enabled:
            return

        records = list(self._memory) if self._memory is not None else []
        if not records:
            return

        self._logger.debug(
            "Learning from %d records (lr=%s, batch_size=%s)",
            len(records),
            self._config.learning_rate,
            self._config.batch_size,
        )
        # Placeholder for actual learning logic that would update model parameters
