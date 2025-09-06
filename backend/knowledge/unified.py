"""Unified knowledge base module.

This module defines :class:`UnifiedKnowledgeBase`, a very small utility that
collects knowledge from heterogeneous sources (science, arts, humanities, etc.)
into a single representation.  The implementation is intentionally lightweight
and serves as a reference for how a more sophisticated integration layer could
work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable


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


@dataclass
class UnifiedKnowledgeBase:
    """Container that stores multiple knowledge sources.

    The class exposes :meth:`add_source` for registering a new knowledge source
    and :meth:`query` for retrieving information across all registered sources.
    """

    sources: Dict[str, KnowledgeSource] = field(default_factory=dict)

    def add_source(self, source: KnowledgeSource) -> None:
        """Register a new knowledge source."""

        self.sources[source.name] = source

    def query(self, concept: str) -> Dict[str, str]:
        """Return a mapping of ``source_name -> description`` for ``concept``.

        The function performs a naive lookup in each source's data mapping.  A
        more capable implementation could involve indexing structures or vector
        search for semantic retrieval.
        """

        results: Dict[str, str] = {}
        for name, source in self.sources.items():
            if concept in source.data:
                results[name] = source.data[concept]
        return results

    def concepts(self) -> Iterable[str]:
        """Iterate over all known concept names."""

        for source in self.sources.values():
            for concept in source.data.keys():
                yield concept
