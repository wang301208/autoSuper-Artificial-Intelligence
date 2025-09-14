from __future__ import annotations

"""Runtime loading and unloading of capability modules.

This module defines :class:`RuntimeModuleManager` which relies on the
:mod:`module_registry` to instantiate capability modules on demand and keeps
track of which modules are currently active.  Modules can be requested or
released dynamically while the system is running, allowing agents to adapt to
new goals without restarting the entire application.
"""

from typing import Any, Dict, Iterable, List

from .module_registry import available_modules, get_module

try:  # Import the common module interface if available
    from modules.interface import ModuleInterface
except Exception:  # pragma: no cover - modules package may not be present
    ModuleInterface = None  # type: ignore

try:  # Optional – the manager can operate without an EventBus
    from events import EventBus  # type: ignore
except Exception:  # pragma: no cover - events module may not be available
    EventBus = None  # type: ignore


class RuntimeModuleManager:
    """Manage dynamic loading/unloading of capability modules."""

    def __init__(self, event_bus: EventBus | None = None) -> None:
        self._loaded: Dict[str, Any] = {}
        self._bus = event_bus
        if self._bus is not None:
            self._bus.subscribe("module.request", self._on_request)
            self._bus.subscribe("module.release", self._on_release)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    async def _on_request(self, event: Dict[str, Any]) -> None:
        name = event.get("module")
        if not isinstance(name, str):
            return
        self.load(name)
        if self._bus:
            self._bus.publish("module.loaded", {"module": name})

    async def _on_release(self, event: Dict[str, Any]) -> None:
        name = event.get("module")
        if not isinstance(name, str):
            return
        try:
            self.unload(name)
        except KeyError:
            return
        if self._bus:
            self._bus.publish("module.unloaded", {"module": name})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(self, name: str, *args, **kwargs) -> Any:
        """Load ``name`` if not already loaded and return the module instance."""
        if name in self._loaded:
            return self._loaded[name]
        module = get_module(name, *args, **kwargs)

        # Resolve dependencies for ModuleInterface implementations
        if ModuleInterface is not None and isinstance(module, ModuleInterface):
            for dep in getattr(module, "dependencies", []):
                self.load(dep)
            module.initialize()

        self._loaded[name] = module
        return module

    def unload(self, name: str) -> None:
        """Unload a previously loaded module."""
        module = self._loaded.pop(name)
        # Gracefully shut down modules. If a module implements ModuleInterface
        # we call its explicit ``shutdown`` hook; otherwise fall back to common
        # cleanup method names.
        if ModuleInterface is not None and isinstance(module, ModuleInterface):
            try:
                module.shutdown()
            except Exception:  # pragma: no cover - best effort
                pass
        else:
            for method in ("shutdown", "close", "stop"):
                func = getattr(module, method, None)
                if callable(func):
                    try:
                        func()
                    except Exception:  # pragma: no cover - best effort
                        pass
                    break

    def update(self, required: Iterable[str]) -> Dict[str, Any]:
        """Ensure ``required`` modules are loaded and drop others.

        Parameters
        ----------
        required:
            Iterable of module names needed for upcoming work. Only names that
            appear in :func:`available_modules` are considered.
        """
        available = set(available_modules())
        needed = {name for name in required if name in available}
        for name in list(self._loaded.keys()):
            if name not in needed:
                self.unload(name)
        return {name: self.load(name) for name in needed}

    def loaded_modules(self) -> List[str]:
        """Return a list of names for currently loaded modules."""
        return list(self._loaded.keys())
