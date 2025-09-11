from __future__ import annotations

"""Protocol definitions for reasoning plugins."""

from typing import Iterable, Protocol, Tuple


class KnowledgeSource(Protocol):
    """Source of information for reasoning."""

    def query(self, statement: str) -> Iterable[str]:
        """Return evidence relevant to ``statement``."""


class Solver(Protocol):
    """Performs inference given a statement and supporting evidence."""

    def infer(self, statement: str, evidence: Iterable[str]) -> Tuple[str, float]:
        """Return a conclusion and its associated probability/confidence."""
