"""Simple in-memory algorithm storage."""

from collections.abc import Callable
from typing import Dict, List


class AlgorithmLibrary:
    """Stores callables representing algorithms keyed by name."""

    def __init__(self) -> None:
        self._algorithms: Dict[str, Callable] = {}

    def add(self, name: str, algorithm: Callable) -> None:
        """Add an algorithm under a unique name."""
        if name in self._algorithms:
            raise ValueError(f"Algorithm '{name}' already exists")
        self._algorithms[name] = algorithm

    def get(self, name: str) -> Callable:
        """Retrieve an algorithm by name."""
        return self._algorithms[name]

    def list(self) -> List[str]:
        """List the names of stored algorithms."""
        return list(self._algorithms.keys())

    def remove(self, name: str) -> None:
        """Remove a stored algorithm."""
        del self._algorithms[name]
