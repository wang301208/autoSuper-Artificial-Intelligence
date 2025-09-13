from __future__ import annotations

"""Utilities for storing visual observations in the world model.

This module provides a very small in-memory store that keeps track of
visual observations associated with agents.  Observations can be raw image
"tensors" (``torch.Tensor`` or ``numpy.ndarray``) or any pre-computed feature
vectors such as those produced by CLIP.  The store makes no assumptions about
 the underlying type and simply keeps the provided objects so that callers can
retrieve them later.
"""

from typing import Any, Dict, Optional


class VisionStore:
    """In-memory storage for visual observations."""

    def __init__(self) -> None:
        self._images: Dict[str, Any] = {}
        self._features: Dict[str, Any] = {}
        self._vit_features: Dict[str, Any] = {}

    def ingest(
        self,
        agent_id: str,
        image: Optional[Any] = None,
        features: Optional[Any] = None,
        vit_features: Optional[Any] = None,
    ) -> None:
        """Store an observation for ``agent_id``.

        Parameters
        ----------
        agent_id:
            Identifier of the agent that produced the observation.
        image:
            Raw image tensor or array.  The store does not modify the object and
            merely keeps a reference to it.
        features:
            Feature representation of the observation (for example CLIP
            embeddings).
        vit_features:
            Feature representation produced by a Vision Transformer (ViT)
            model.
        """

        if image is not None:
            self._images[agent_id] = image
        if features is not None:
            self._features[agent_id] = features
        if vit_features is not None:
            self._vit_features[agent_id] = vit_features

    def get(self, agent_id: str) -> Dict[str, Any]:
        """Return the latest observation for ``agent_id``."""

        return {
            "image": self._images.get(agent_id),
            "features": self._features.get(agent_id),
            "vit_features": self._vit_features.get(agent_id),
        }

    def all(self) -> Dict[str, Dict[str, Any]]:
        """Return a snapshot of all stored observations."""

        keys = set(self._images) | set(self._features) | set(self._vit_features)
        return {
            agent: {
                "image": self._images.get(agent),
                "features": self._features.get(agent),
                "vit_features": self._vit_features.get(agent),
            }
            for agent in keys
        }


__all__ = ["VisionStore"]
