"""Simple cross-modal transformer for modality alignment and fusion.

This module provides a tiny stand-in implementation that projects each
modality to a common embedding space and averages them.  It is designed
for testing and illustrative purposes rather than performance.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class CrossModalTransformer:
    """Align and fuse multiple sensory modalities into one vector.

    Parameters
    ----------
    output_dim:
        Dimensionality of the shared representation space.  All modalities
        are projected to this size before fusion.
    """

    def __init__(self, output_dim: int = 16) -> None:
        self.output_dim = int(output_dim)

    def _project(self, x: np.ndarray) -> np.ndarray:
        """Project an input modality to the shared representation space.

        The projection here is intentionally simple: we compute the mean of
        the modality and repeat it ``output_dim`` times.  This keeps the
        implementation deterministic and lightweight while allowing tests to
        verify behaviour.
        """

        arr = np.asarray(x, dtype=float).reshape(-1)
        if arr.size == 0:
            raise ValueError("modality input is empty")
        mean = arr.mean()
        return np.full(self.output_dim, mean, dtype=float)

    def fuse(self, modalities: Sequence[np.ndarray]) -> np.ndarray:
        """Fuse multiple modalities into a unified representation.

        Parameters
        ----------
        modalities:
            Sequence of modality arrays to be fused.  Each modality is first
            aligned to the shared space and then averaged.

        Returns
        -------
        np.ndarray
            The fused representation of shape ``(output_dim,)``.
        """

        if not modalities or any(m is None for m in modalities):
            raise ValueError("modalities must be provided")
        aligned = [self._project(m) for m in modalities]
        return np.mean(aligned, axis=0)


__all__ = ["CrossModalTransformer"]
