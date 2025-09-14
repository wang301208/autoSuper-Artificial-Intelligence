"""Engine for fusing multiple sensory modalities.

The :class:`MultimodalFusionEngine` provides a tiny abstraction around a
``CrossModalTransformer``.  It exposes a dependency injection interface so
that different transformer implementations can be supplied in tests or by
other parts of the system.
"""

from __future__ import annotations

from typing import Dict, Sequence, Optional

import numpy as np

from .cross_modal_transformer import CrossModalTransformer


class MultimodalFusionEngine:
    """Fuse data from arbitrary modalities into a shared representation using attention."""

    def __init__(self, transformer: Optional[CrossModalTransformer] = None) -> None:
        self._transformer = transformer or CrossModalTransformer()

    def set_transformer(self, transformer: CrossModalTransformer) -> None:
        """Inject a different :class:`CrossModalTransformer` instance."""

        self._transformer = transformer

    def fuse_sensory_modalities(self, **modalities: np.ndarray) -> np.ndarray:
        """Return a unified representation of the provided modalities.

        Parameters
        ----------
        **modalities:
            Named modality arrays (e.g. ``visual``, ``auditory``, ``tactile``,
            ``smell`` or ``text``).  At least one modality must be supplied.
            Each modality is aligned via the configured transformer and fused
            using a simple attention mechanism that weights modalities by the
            magnitude of their aligned representations.
        """

        if not modalities:
            raise ValueError("at least one modality must be provided")

        aligned = self._align_modalities(modalities)
        weights = self._attention(aligned)
        return np.average(aligned, axis=0, weights=weights)

    def _align_modalities(self, modalities: Dict[str, np.ndarray]) -> Sequence[np.ndarray]:
        """Project modalities into the shared representation space."""

        return [self._transformer.project(m) for m in modalities.values()]

    @staticmethod
    def _attention(aligned: Sequence[np.ndarray]) -> np.ndarray:
        """Compute attention weights for a sequence of aligned modalities."""

        scores = np.array([np.linalg.norm(a) for a in aligned], dtype=float)
        if np.allclose(scores, 0):
            return np.full(len(aligned), 1.0 / len(aligned))
        scores -= np.max(scores)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum()


__all__ = ["MultimodalFusionEngine"]
