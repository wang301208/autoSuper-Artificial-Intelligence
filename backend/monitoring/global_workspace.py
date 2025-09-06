from __future__ import annotations

"""Shared global workspace for broadcasting state between modules."""

from typing import Any, Dict


class GlobalWorkspace:
    """Registry that enables modules to share state and attention."""

    def __init__(self) -> None:
        self._modules: Dict[str, Any] = {}
        self._state: Dict[str, Any] = {}
        self._attention: Dict[str, float] = {}

    # ------------------------------------------------------------------
    def register_module(self, name: str, module: Any) -> None:
        """Register *module* under *name* in the workspace."""
        self._modules[name] = module

    def broadcast(self, sender: str, state: Any, attention: float | None = None) -> None:
        """Broadcast *state* and optional *attention* from *sender* to all other modules."""
        self._state[sender] = state
        if attention is not None:
            self._attention[sender] = float(attention)
        for name, module in self._modules.items():
            if name == sender:
                continue
            handler = getattr(module, "receive_broadcast", None)
            if callable(handler):
                handler(sender, state, attention)

    # ------------------------------------------------------------------
    def state(self, name: str) -> Any:
        """Return the last state published by *name*."""
        return self._state.get(name)

    def attention(self, name: str) -> float | None:
        """Return the last attention weight published by *name*."""
        return self._attention.get(name)


# Global workspace instance

global_workspace = GlobalWorkspace()
