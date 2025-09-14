"""Adapter for evolution module implementing the common ModuleInterface."""

from __future__ import annotations

from modules.interface import ModuleInterface


class EvolutionModule(ModuleInterface):
    """Expose evolution capabilities via ModuleInterface."""

    dependencies: list[str] = []

    def __init__(self) -> None:
        self.initialized = False

    def initialize(self) -> None:  # pragma: no cover - trivial
        self.initialized = True

    def shutdown(self) -> None:  # pragma: no cover - trivial
        self.initialized = False
