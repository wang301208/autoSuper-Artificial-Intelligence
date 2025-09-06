"""Data structures for representing multimodal concepts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ConceptNode:
    """A node in a concept graph with embeddings for multiple modalities."""

    id: str
    label: str
    modalities: Dict[str, List[float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConceptRelation:
    """A typed edge between two concept nodes."""

    source: str
    target: str
    relation: str
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
