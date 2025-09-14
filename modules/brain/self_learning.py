from __future__ import annotations

"""Self-learning brain module with curiosity-driven updates."""

from dataclasses import dataclass, field
from typing import Any, Dict, Set

try:  # pragma: no cover - fallback if ML dependencies are missing
    from backend.ml.experience_collector import ActiveCuriositySelector
except Exception:  # pragma: no cover - lightweight stand-in for tests
    class ActiveCuriositySelector:  # type: ignore[override]
        """Minimal curiosity selector used when full backend is unavailable."""

        def __init__(
            self, reward_threshold: float = 0.0, novelty_weight: float = 0.5
        ) -> None:
            self.reward_threshold = reward_threshold
            self.novelty_weight = novelty_weight
            self.seen_states: Set[str] = set()
            self.avg_reward = 0.0
            self.count = 0

        def consider(self, sample: Dict[str, Any]) -> bool:
            self.count += 1
            r = sample["reward"]
            self.avg_reward += (r - self.avg_reward) / self.count
            novelty = 0.0 if sample["state"] in self.seen_states else 1.0
            curiosity = self.novelty_weight * novelty + (1 - self.novelty_weight) * max(
                0.0, r - self.avg_reward
            )
            if curiosity > self.reward_threshold:
                self.seen_states.add(sample["state"])
                return True
            return False

try:  # pragma: no cover - use full implementation if available
    from backend.world_model import WorldModel
except Exception:  # pragma: no cover - simplified model for testing
    class WorldModel:  # type: ignore[override]
        """Minimal world model tracking resource usage with EWMA."""

        def __init__(self, alpha: float = 0.5) -> None:
            self.alpha = alpha
            self._predictions: Dict[str, Dict[str, float]] = {}

        def update_resources(self, agent_id: str, usage: Dict[str, float]) -> None:
            prev = self._predictions.get(agent_id)
            if prev is None:
                self._predictions[agent_id] = {
                    "cpu": usage.get("cpu", 0.0),
                    "memory": usage.get("memory", 0.0),
                }
            else:
                self._predictions[agent_id] = {
                    "cpu": self.alpha * usage.get("cpu", 0.0)
                    + (1 - self.alpha) * prev.get("cpu", 0.0),
                    "memory": self.alpha * usage.get("memory", 0.0)
                    + (1 - self.alpha) * prev.get("memory", 0.0),
                }

        def predict(self, agent_id: str) -> Dict[str, float]:
            return self._predictions.get(agent_id, {"cpu": 0.0, "memory": 0.0})


@dataclass
class SelfLearningBrain:
    """Combine curiosity-based selection with a simple world model.

    The brain keeps a lightweight memory of novel states. When an interaction
    sample is considered interesting by :class:`ActiveCuriositySelector`, the
    sample is stored and the :class:`WorldModel` is updated with the observed
    resource usage, correcting its predictions.  The updated prediction can be
    used as an improved policy for future interactions.
    """

    world_model: WorldModel = field(default_factory=WorldModel)
    selector: ActiveCuriositySelector = field(default_factory=ActiveCuriositySelector)
    memory: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def curiosity_driven_learning(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """Update world model based on curiosity and return new predictions.

        Parameters
        ----------
        sample:
            Mapping with at least ``state``, ``reward``, ``agent_id`` and
            ``usage`` (a dict of resource metrics).  Only samples deemed
            interesting by the selector are used to update the model.

        Returns
        -------
        dict
            The world model's prediction for ``agent_id`` after potential
            updates.
        """

        agent_id = sample.get("agent_id", sample["state"])

        if self.selector.consider(sample):
            # store novel sample for memory
            self.memory[sample["state"]] = sample
            usage = sample.get("usage", {})
            if usage:
                # correct world model prediction with observed usage
                self.world_model.update_resources(agent_id, usage)

        return self.world_model.predict(agent_id)


__all__ = ["SelfLearningBrain"]
