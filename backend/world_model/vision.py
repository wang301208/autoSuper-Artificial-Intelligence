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

import torch
from torch import nn


class CrossModalAttention(nn.Module):
    """Simple cross-modal attention module.

    The module attends between visual and textual features and produces a
    unified representation by averaging bidirectional attention outputs.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8) -> None:
        if embed_dim % num_heads != 0:
            num_heads = 1
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, vision_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        if vision_feat.dim() == 1:
            vision_feat = vision_feat.unsqueeze(0).unsqueeze(0)
        elif vision_feat.dim() == 2:
            vision_feat = vision_feat.unsqueeze(0)
        if text_feat.dim() == 1:
            text_feat = text_feat.unsqueeze(0).unsqueeze(0)
        elif text_feat.dim() == 2:
            text_feat = text_feat.unsqueeze(0)
        text_attended, _ = self.attn(text_feat, vision_feat, vision_feat)
        vision_attended, _ = self.attn(vision_feat, text_feat, text_feat)
        unified = (text_attended.mean(dim=1) + vision_attended.mean(dim=1)) / 2
        return unified.squeeze(0)


class VisionStore:
    """In-memory storage for visual observations."""

    def __init__(self) -> None:
        self._images: Dict[str, Any] = {}
        self._features: Dict[str, Any] = {}
        self._vit_features: Dict[str, Any] = {}
        self._text: Dict[str, Any] = {}
        self._unified: Dict[str, Any] = {}
        self._attn: Optional[CrossModalAttention] = None

    def ingest(
        self,
        agent_id: str,
        image: Optional[Any] = None,
        features: Optional[Any] = None,
        vit_features: Optional[Any] = None,
        text: Optional[Any] = None,
    ) -> Optional[torch.Tensor]:
        """Store an observation for ``agent_id`` and optionally compute fusion.

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
        text:
            Optional textual embedding aligned with the visual input.
        """

        if image is not None:
            self._images[agent_id] = image
        if features is not None:
            self._features[agent_id] = features
        if vit_features is not None:
            self._vit_features[agent_id] = vit_features
        if text is not None:
            self._text[agent_id] = text

        unified: Optional[torch.Tensor] = None
        vision_feat = features if features is not None else vit_features
        if text is not None and vision_feat is not None:
            vision_tensor = torch.as_tensor(vision_feat)
            text_tensor = torch.as_tensor(text)
            if self._attn is None:
                self._attn = CrossModalAttention(vision_tensor.shape[-1])
            unified = self._attn(vision_tensor, text_tensor)
            self._unified[agent_id] = unified

        return unified

    def get(self, agent_id: str) -> Dict[str, Any]:
        """Return the latest observation for ``agent_id``."""

        return {
            "image": self._images.get(agent_id),
            "features": self._features.get(agent_id),
            "vit_features": self._vit_features.get(agent_id),
            "text": self._text.get(agent_id),
            "unified": self._unified.get(agent_id),
        }

    def all(self) -> Dict[str, Dict[str, Any]]:
        """Return a snapshot of all stored observations."""

        keys = (
            set(self._images)
            | set(self._features)
            | set(self._vit_features)
            | set(self._text)
            | set(self._unified)
        )
        return {
            agent: {
                "image": self._images.get(agent),
                "features": self._features.get(agent),
                "vit_features": self._vit_features.get(agent),
                "text": self._text.get(agent),
                "unified": self._unified.get(agent),
            }
            for agent in keys
        }


__all__ = ["VisionStore", "CrossModalAttention"]
