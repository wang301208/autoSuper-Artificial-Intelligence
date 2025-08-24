"""Learning utilities for adapting the agent from past experiences."""
from __future__ import annotations

import logging
from typing import Iterable, Dict

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

    def learn_from_experience(self) -> Dict[str, float]:
        """Read past interactions from memory and update the model.

        Returns:
            A mapping of command names to updated priority weights. Commands with
            more successful executions receive higher weights.
        """
        if not self._config.enabled:
            return {}

        records = list(self._memory) if self._memory is not None else []
        if not records:
            return {}

        self._logger.debug(
            "Learning from %d records (lr=%s, batch_size=%s)",
            len(records),
            self._config.learning_rate,
            self._config.batch_size,
        )

        stats: dict[str, dict[str, int]] = {}
        for episode in records:
            if not getattr(episode, "result", None):
                continue
            cmd_name = episode.action.name
            stats.setdefault(cmd_name, {"success": 0, "total": 0})
            stats[cmd_name]["total"] += 1
            if getattr(episode.result, "status", None) == "success":
                stats[cmd_name]["success"] += 1

        weights = {
            name: (values["success"] / values["total"])
            if values["total"]
            else 0
            for name, values in stats.items()
        }

        self._logger.debug("Learned command weights: %s", weights)
        return weights
