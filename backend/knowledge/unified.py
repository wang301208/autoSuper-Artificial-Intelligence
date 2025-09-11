"""Unified knowledge base module.

This module defines :class:`UnifiedKnowledgeBase`, a very small utility that
collects knowledge from heterogeneous sources (science, arts, humanities, etc.)
into a single representation.  The implementation is intentionally lightweight
and serves as a reference for how a more sophisticated integration layer could
work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:  # Optional dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - handled gracefully
    SentenceTransformer = None  # type: ignore

from ..memory.long_term import LongTermMemory
from .vector_store import LocalVectorStore


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
    vector_store: Optional[LocalVectorStore] = None

    # ------------------------------------------------------------------
    # Internal helpers for persistence
    # ------------------------------------------------------------------
    def _embed_key(self, source_name: str, concept: str) -> str:
        return f"{source_name}:{concept}"

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
            key = self._embed_key(source.name, concept)
            stored = self.memory.get_embedding(key) if self.memory else None
            emb: List[float]
            if stored is not None:
                emb = stored[0]
            else:
                emb = self.embedder.encode(description)
                if isinstance(emb, np.ndarray):
                    emb = emb.tolist()
                if self.memory is not None:
                    self.memory.add_embedding(
                        key, emb, {"source": source.name, "concept": concept}
                    )
            source.embeddings[concept] = np.asarray(emb, dtype=float)
            if self.vector_store is None:
                self.vector_store = LocalVectorStore(len(emb))
            self.vector_store.add(
                source.embeddings[concept],
                {
                    "source": source.name,
                    "concept": concept,
                    "description": description,
                },
            )

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

        if self.embedder is None or self.vector_store is None:
            return {}

        query_emb = self.embedder.encode(concept)
        if isinstance(query_emb, list):
            query_emb = np.asarray(query_emb, dtype=float)
        query_emb = np.asarray(query_emb, dtype=float)

        hits = self.vector_store.search(query_emb, top_k=top_k)
        results: Dict[str, str] = {}
        for hit in hits:
            source_name = hit.get("source", "")
            concept_name = hit.get("concept", "")
            description = hit.get("description", "")
            results[f"{source_name}:{concept_name}"] = description
        return results

    def concepts(self) -> Iterable[str]:
        """Iterate over all known concept names."""

        for source in self.sources.values():
            for concept in source.data.keys():
                yield concept
