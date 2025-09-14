"""Engine for fusing multiple sensory modalities.

The :class:`MultimodalFusionEngine` provides a tiny abstraction around a
``CrossModalTransformer``.  It exposes a dependency injection interface so
that different transformer implementations can be supplied in tests or by
other parts of the system.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .cross_modal_transformer import CrossModalTransformer


class MultimodalFusionEngine:
    """Fuse visual, auditory and tactile data into a shared representation."""

    def __init__(self, transformer: Optional[CrossModalTransformer] = None) -> None:
        self._transformer = transformer or CrossModalTransformer()

    def set_transformer(self, transformer: CrossModalTransformer) -> None:
        """Inject a different :class:`CrossModalTransformer` instance."""

        self._transformer = transformer

    def fuse_sensory_modalities(
        self, visual: np.ndarray, auditory: np.ndarray, tactile: np.ndarray
    ) -> np.ndarray:
        """Return a unified representation of the provided modalities.

        Parameters
        ----------
        visual, auditory, tactile:
            Arrays containing the sensory data.  All three modalities must be
            supplied.  Each is aligned via the configured transformer and then
            fused into a single representation.
        """

        modalities = [visual, auditory, tactile]
        if any(m is None for m in modalities):
            raise ValueError("all modalities must be provided")
        return self._transformer.fuse(modalities)


__all__ = ["MultimodalFusionEngine"]
