"""Lightweight local vector store for semantic search.

This module implements a minimal wrapper around FAISS_ for storing and
retrieving embedding vectors.  If FAISS is not available, it falls back to a
simple numpy based index.  The implementation is intentionally compact yet
sufficient for small projects and unit tests.

Usage
-----
>>> store = LocalVectorStore(3)
>>> store.add([0.1, 0.2, 0.3], {"id": 1})
>>> store.search([0.1, 0.2, 0.3])
[{"id": 1}]

.. _FAISS: https://github.com/facebookresearch/faiss
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:  # Optional dependency
    import faiss  # type: ignore
except Exception:  # pragma: no cover - handled gracefully
    faiss = None  # type: ignore


@dataclass
class LocalVectorStore:
    """Small in-memory vector index with optional FAISS acceleration."""

    dimension: int
    use_faiss: bool = True
    _faiss_index: Any | None = field(init=False, default=None)
    _embeddings: List[np.ndarray] = field(init=False, default_factory=list)
    _meta: List[Dict[str, Any]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.use_faiss = self.use_faiss and faiss is not None
        if self.use_faiss:
            # Inner product is equivalent to cosine similarity on normalised vectors
            self._faiss_index = faiss.IndexFlatIP(self.dimension)

    # ------------------------------------------------------------------
    def add(self, vector: Iterable[float], metadata: Dict[str, Any]) -> None:
        """Add a vector with associated metadata to the store."""

        vec = np.asarray(vector, dtype="float32")
        if vec.shape != (self.dimension,):
            raise ValueError("Vector has wrong dimensions")
        if self.use_faiss and self._faiss_index is not None:
            self._faiss_index.add(vec.reshape(1, -1))
        else:
            self._embeddings.append(vec)
        self._meta.append(metadata)

    # ------------------------------------------------------------------
    def search(self, query: Iterable[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Return metadata for vectors most similar to ``query``."""

        q = np.asarray(query, dtype="float32")
        if q.shape != (self.dimension,):
            raise ValueError("Query has wrong dimensions")

        if self.use_faiss and self._faiss_index is not None and self._faiss_index.ntotal > 0:
            _, idx = self._faiss_index.search(q.reshape(1, -1), top_k)
            return [self._meta[i] for i in idx[0] if i < len(self._meta)]

        if not self._embeddings:
            return []
        embs = np.vstack(self._embeddings)
        norms = np.linalg.norm(embs, axis=1) * np.linalg.norm(q)
        norms = np.where(norms == 0, 1e-10, norms)
        sims = embs @ q / norms
        best_idx = np.argsort(sims)[::-1][:top_k]
        return [self._meta[i] for i in best_idx]
