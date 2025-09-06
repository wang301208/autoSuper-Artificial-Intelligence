from __future__ import annotations

from typing import Any, Callable, Dict, List

_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_module(name: str, factory: Callable[..., Any]) -> None:
    """Register a module factory under ``name``."""
    _REGISTRY[name] = factory


def get_module(name: str, *args, **kwargs) -> Any:
    """Instantiate a registered module."""
    if name not in _REGISTRY:
        raise KeyError(f"Module '{name}' is not registered")
    return _REGISTRY[name](*args, **kwargs)


def available_modules() -> List[str]:
    """Return a list of registered module names."""
    return list(_REGISTRY.keys())


def combine_modules(names: List[str]) -> List[Any]:
    """Instantiate multiple modules by name."""
    return [get_module(name) for name in names]
