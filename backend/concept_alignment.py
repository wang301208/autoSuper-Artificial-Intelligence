"""Utilities for aligning text embeddings with knowledge graph entities."""
from __future__ import annotations

from functools import lru_cache
from math import sqrt
from typing import Dict, List, Tuple

import yaml

from capability.librarian import Librarian
from modules.common import ConceptNode


class ConceptAligner:
    """Align query embeddings to knowledge graph concept nodes."""

    def __init__(
        self,
        librarian: Librarian,
        entities: Dict[str, ConceptNode],
        encoders: Dict[str, str] | None = None,
    ) -> None:
        self.librarian = librarian
        self.entities = entities
        self.encoders = encoders or {}

    @classmethod
    def from_config(
        cls, librarian: Librarian, entities: Dict[str, ConceptNode], config_path: str
    ) -> "ConceptAligner":
        """Construct an aligner from a YAML configuration file."""
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        encoders = data.get("encoders", {})
        return cls(librarian=librarian, entities=entities, encoders=encoders)

    @lru_cache(maxsize=128)
    def _cached_search(
        self, embedding_key: Tuple[float, ...], n_results: int, vector_type: str
    ) -> Tuple[str, ...]:
        return tuple(
            self.librarian.search(
                list(embedding_key),
                n_results=n_results,
                vector_type=vector_type,
                return_content=False,
            )
        )

    def align(
        self, query_embedding: List[float], n_results: int = 5, vector_type: str = "text"
    ) -> List[ConceptNode]:
        """Return concept nodes from the knowledge graph most similar to the query."""
        entity_ids = self._cached_search(tuple(query_embedding), n_results, vector_type)
        results: List[ConceptNode] = []
        for entity_id in entity_ids:
            node = self.entities.get(entity_id)
            if not node:
                continue
            embedding = node.modalities.get(vector_type)
            if embedding is None:
                continue
            similarity = self._cosine_similarity(query_embedding, embedding)
            node.metadata["similarity"] = similarity
            results.append(node)
        results.sort(key=lambda n: n.metadata.get("similarity", 0.0), reverse=True)
        return results

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sqrt(sum(x * x for x in a))
        norm_b = sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
