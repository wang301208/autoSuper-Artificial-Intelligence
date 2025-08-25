"""Event bus abstraction for AutoGPT.

This module defines a minimal publish/subscribe interface and exposes helper
functions for interacting with a global event bus instance. The default
implementation is an in-memory bus, but alternative backends such as Redis can
be plugged in to enable coordination across multiple hosts.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

__all__ = [
    "EventBus",
    "InMemoryEventBus",
    "event_bus",
    "create_event_bus",
    "get_event_bus",
    "set_event_bus",
    "publish",
    "subscribe",
    "unsubscribe",
]


class EventBus(ABC):
    """Simple publish/subscribe interface used by AutoGPT."""

    @abstractmethod
    def publish(self, topic: str, event: Dict[str, Any]) -> None:
        """Publish *event* on *topic*."""

    @abstractmethod
    def subscribe(
        self, topic: str, handler: Callable[[Dict[str, Any]], None]
    ) -> Callable[[], None]:
        """Subscribe *handler* to events on *topic* and return a cancel function."""

    @abstractmethod
    def unsubscribe(
        self, topic: str, handler: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Remove *handler* subscription from *topic*."""


class InMemoryEventBus(EventBus):
    """Simple in-memory pub/sub event bus."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
        self._lock = threading.Lock()

    def publish(self, topic: str, event: Dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._subscribers.get(topic, []))
        for handler in subscribers:
            try:
                handler(event)
            except Exception:
                # Don't allow one bad handler to break the others
                pass

    def subscribe(
        self, topic: str, handler: Callable[[Dict[str, Any]], None]
    ) -> Callable[[], None]:
        with self._lock:
            self._subscribers.setdefault(topic, []).append(handler)
        return lambda: self.unsubscribe(topic, handler)

    def unsubscribe(
        self, topic: str, handler: Callable[[Dict[str, Any]], None]
    ) -> None:
        with self._lock:
            handlers = self._subscribers.get(topic)
            if handlers and handler in handlers:
                handlers.remove(handler)
                if not handlers:
                    del self._subscribers[topic]


event_bus: EventBus = InMemoryEventBus()


def set_event_bus(bus: EventBus) -> None:
    """Set the global event bus instance."""

    global event_bus
    event_bus = bus


def get_event_bus() -> EventBus:
    """Return the currently configured global event bus."""

    return event_bus


def create_event_bus(backend: str, **kwargs: Any) -> EventBus:
    """Create an event bus for *backend*.

    Args:
        backend: Name of the backend to use. ``"inmemory"`` selects the built-in
            bus. ``"redis"`` selects :class:`RedisEventBus`.
        **kwargs: Backend specific options.
    """

    backend = (backend or "inmemory").lower()
    if backend == "redis":
        from .redis_bus import RedisEventBus

        return RedisEventBus(**kwargs)
    return InMemoryEventBus()


def publish(topic: str, event: Dict[str, Any]) -> None:
    """Publish *event* on *topic* using the configured event bus."""

    event_bus.publish(topic, event)


def subscribe(topic: str, handler: Callable[[Dict[str, Any]], None]) -> Callable[[], None]:
    """Subscribe *handler* to events published on *topic*."""

    return event_bus.subscribe(topic, handler)


def unsubscribe(topic: str, handler: Callable[[Dict[str, Any]], None]) -> None:
    """Remove *handler* subscription from *topic*."""

    event_bus.unsubscribe(topic, handler)

