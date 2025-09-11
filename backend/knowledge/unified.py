"""Unified knowledge base module.

This module defines :class:`UnifiedKnowledgeBase`, a very small utility that
collects knowledge from heterogeneous sources (science, arts, humanities, etc.)
into a single representation.  The implementation is intentionally lightweight
and serves as a reference for how a more sophisticated integration layer could
work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:  # Optional dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - handled gracefully
    SentenceTransformer = None  # type: ignore

from ..memory.long_term import LongTermMemory


@dataclass
class KnowledgeSource:
    """Simple container for a knowledge source.

    Attributes
    ----------
    name:
        Identifier for the knowledge source.
    data:
        Mapping of concept names to descriptions.
    """

    name: str
    data: Dict[str, str]
    embeddings: Optional[Dict[str, np.ndarray]] = None


@dataclass
class UnifiedKnowledgeBase:
    """Container that stores multiple knowledge sources.

    The class exposes :meth:`add_source` for registering a new knowledge source
    and :meth:`query` for retrieving information across all registered sources.
    """

    sources: Dict[str, KnowledgeSource] = field(default_factory=dict)
    embedder: Any | None = None
    memory: Optional[LongTermMemory] = None
    _index_embeddings: np.ndarray = field(init=False, default_factory=lambda: np.empty((0, 0)))
    _index_meta: List[Tuple[str, str, str]] = field(init=False, default_factory=list)

    # ------------------------------------------------------------------
    # Internal helpers for persistence
    # ------------------------------------------------------------------
    def _embed_key(self, source_name: str, concept: str) -> str:
        return f"{source_name}:{concept}"

    def _save_embedding(self, source_name: str, concept: str, embedding: List[float]) -> None:
        if self.memory is None:
            return
        payload = {"key": self._embed_key(source_name, concept), "embedding": embedding}
        self.memory.add("knowledge_embedding", json.dumps(payload))

    def _load_embedding(self, source_name: str, concept: str) -> Optional[List[float]]:
        if self.memory is None:
            return None
        key = self._embed_key(source_name, concept)
        for content in self.memory.get("knowledge_embedding"):
            try:
                payload = json.loads(content)
            except Exception:  # pragma: no cover - defensive
                continue
            if payload.get("key") == key:
                emb = payload.get("embedding")
                if isinstance(emb, list):
                    return emb
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_source(self, source: KnowledgeSource) -> None:
        """Register a new knowledge source and compute embeddings."""

        self.sources[source.name] = source
        if self.embedder is None:
            return
        if source.embeddings is None:
            source.embeddings = {}
        for concept, description in source.data.items():
            emb: Optional[List[float]] = self._load_embedding(source.name, concept)
            if emb is None:
                emb = self.embedder.encode(description)
                # ``SentenceTransformer.encode`` can return numpy arrays or lists
                if isinstance(emb, np.ndarray):
                    emb = emb.tolist()
                self._save_embedding(source.name, concept, emb)
            source.embeddings[concept] = np.asarray(emb, dtype=float)
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        embeddings: List[np.ndarray] = []
        meta: List[Tuple[str, str, str]] = []
        for source_name, source in self.sources.items():
            if not source.embeddings:
                continue
            for concept, emb in source.embeddings.items():
                embeddings.append(np.asarray(emb, dtype=float))
                meta.append((source_name, concept, source.data[concept]))
        if embeddings:
            self._index_embeddings = np.vstack(embeddings)
        else:
            self._index_embeddings = np.empty((0, 0))
        self._index_meta = meta

    def query(self, concept: str, *, semantic: bool = False, top_k: int = 5) -> Dict[str, str]:
        """Retrieve concept descriptions.

        Parameters
        ----------
        concept:
            Concept or query string.
        semantic:
            If ``True`` perform vector search using cosine similarity across all
            indexed concepts.  Otherwise do a naive lookup.
        top_k:
            Number of semantic matches to return.
        """

        if not semantic:
            results: Dict[str, str] = {}
            for name, source in self.sources.items():
                if concept in source.data:
                    results[name] = source.data[concept]
            return results

        if self.embedder is None or self._index_embeddings.size == 0:
            return {}

        query_emb = self.embedder.encode(concept)
        if isinstance(query_emb, list):
            query_emb = np.asarray(query_emb, dtype=float)
        query_emb = np.asarray(query_emb, dtype=float)

        norms = np.linalg.norm(self._index_embeddings, axis=1) * np.linalg.norm(query_emb)
        norms = np.where(norms == 0, 1e-10, norms)
        sims = self._index_embeddings @ query_emb / norms
        best_idx = np.argsort(sims)[::-1][:top_k]

        results: Dict[str, str] = {}
        for idx in best_idx:
            source_name, concept_name, description = self._index_meta[idx]
            results[f"{source_name}:{concept_name}"] = description
        return results

    def concepts(self) -> Iterable[str]:
        """Iterate over all known concept names."""

        for source in self.sources.values():
            for concept in source.data.keys():
                yield concept
