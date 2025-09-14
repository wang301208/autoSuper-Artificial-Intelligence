"""Adapter for brain module implementing the common ModuleInterface."""

from __future__ import annotations

from modules.interface import ModuleInterface


class BrainModule(ModuleInterface):
    """Thin adapter exposing brain functionality via ModuleInterface."""

    # For demonstration purposes the brain depends on the evolution module.
    dependencies = ["evolution"]

    def __init__(self) -> None:
        self.initialized = False

    def initialize(self) -> None:  # pragma: no cover - trivial
        self.initialized = True

    def shutdown(self) -> None:  # pragma: no cover - trivial
        self.initialized = False
