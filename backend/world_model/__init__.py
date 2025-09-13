"""World model module.

Provides a simple in-memory representation of the environment's state and a
learning mechanism for predicting future resource usage. Agents can query and
update this model during execution.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .vision import VisionStore


class WorldModel:
    """Maintain a structured state of the environment.

    The model tracks tasks, resource usage of agents and their recent actions.
    A lightweight learning component keeps exponentially-weighted moving
    averages of resource usage, which serve as predictions for future usage.
    """

    def __init__(self, alpha: float = 0.5) -> None:
        """Create a new world model.

        Args:
            alpha: Smoothing factor for the moving average used in learning
                resource usage patterns. ``0 < alpha <= 1``.
        """

        self.alpha = alpha
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.resources: Dict[str, Dict[str, float]] = {}
        self.actions: List[Dict[str, str]] = []
        self._predictions: Dict[str, Dict[str, float]] = {}
        self.vision = VisionStore()
        self.multimodal: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # State management APIs
    # ------------------------------------------------------------------
    def add_task(self, task_id: str, metadata: Dict[str, Any]) -> None:
        """Add or update a task in the world model."""

        self.tasks[task_id] = metadata

    def update_resources(self, agent_id: str, usage: Dict[str, float]) -> None:
        """Update the current resource usage for an agent.

        This method also updates the internal prediction for the agent using an
        exponentially-weighted moving average (EWMA) based on past interactions.
        """

        self.resources[agent_id] = usage

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

    def record_action(self, agent_id: str, action: str) -> None:
        """Record an action performed by an agent."""

        self.actions.append({"agent_id": agent_id, "action": action})

    def get_state(self) -> Dict[str, Any]:
        """Return a snapshot of the current world state."""

        return {
            "tasks": dict(self.tasks),
            "resources": {k: dict(v) for k, v in self.resources.items()},
            "actions": list(self.actions),
            "vision": self.vision.all(),
            "multimodal": {k: v for k, v in self.multimodal.items()},
        }

    def add_visual_observation(
        self,
        agent_id: str,
        image: Any | None = None,
        features: Any | None = None,
        vit_features: Any | None = None,
        text: Any | None = None,
    ) -> None:
        """Store visual data for ``agent_id``.

        Parameters
        ----------
        agent_id:
            Identifier of the agent that produced the observation.
        image:
            Optional raw image tensor or array.
        features:
            Optional feature vector representing the image.
        vit_features:
            Optional feature vector produced by a ViT model.
        text:
            Optional textual embedding associated with the observation.
        """

        unified = self.vision.ingest(
            agent_id,
            image=image,
            features=features,
            vit_features=vit_features,
            text=text,
        )
        if unified is not None:
            self.multimodal[agent_id] = unified

    def get_visual_observation(self, agent_id: str) -> Dict[str, Any]:
        """Retrieve the latest visual observation for ``agent_id``."""

        return self.vision.get(agent_id)

    def get_unified_representation(self, agent_id: str) -> Any:
        """Retrieve the unified multimodal representation for ``agent_id``."""

        return self.multimodal.get(agent_id)

    # ------------------------------------------------------------------
    # Prediction APIs
    # ------------------------------------------------------------------
    def predict(self, agent_id: Optional[str] = None) -> Dict[str, float]:
        """Predict resource usage.

        Args:
            agent_id: If provided, return predictions specific to this agent.
                Otherwise, return average predictions across all agents.
        """

        if agent_id is not None:
            return self._predictions.get(agent_id, {"cpu": 0.0, "memory": 0.0})

        if not self._predictions:
            return {"avg_cpu": 0.0, "avg_memory": 0.0}

        total_cpu = sum(p.get("cpu", 0.0) for p in self._predictions.values())
        total_mem = sum(p.get("memory", 0.0) for p in self._predictions.values())
        count = len(self._predictions)
        return {"avg_cpu": total_cpu / count, "avg_memory": total_mem / count}


__all__ = ["WorldModel", "VisionStore"]

