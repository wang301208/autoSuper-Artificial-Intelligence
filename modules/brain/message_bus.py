"""Neural message bus for communication between brain regions."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List


class EventDispatcher:
    """Dispatches events to registered handlers."""

    def __init__(self) -> None:
        self._listeners: Dict[str, List[Callable[[Any], None]]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: Callable[[Any], None]) -> None:
        """Register a handler for a specific event type."""
        self._listeners[event_type].append(handler)

    def dispatch(self, event_type: str, data: Any) -> None:
        """Send data to all handlers of an event type."""
        for handler in list(self._listeners.get(event_type, [])):
            try:
                handler(data)
            except Exception:
                # Isolate faulty handlers so others can proceed
                continue


class CircuitBreaker:
    """Simple circuit breaker to isolate failing handlers."""

    def __init__(self, threshold: int = 3) -> None:
        self.threshold = threshold
        self.failures = 0
        self.open = False

    def call(self, func: Callable[[Any], None], data: Any) -> None:
        if self.open:
            return
        try:
            func(data)
            self.failures = 0
        except Exception:
            self.failures += 1
            if self.failures >= self.threshold:
                self.open = True
            raise


class MessageRouter:
    """Routes events to brain regions via an EventDispatcher."""

    def __init__(self, dispatcher: EventDispatcher) -> None:
        self.dispatcher = dispatcher
        self._breakers: Dict[Callable[[Any], None], CircuitBreaker] = {}

    def register(
        self,
        region: str,
        event_type: str,
        handler: Callable[[Any], None],
        threshold: int = 3,
    ) -> None:
        """Register a region's handler for an event type."""
        breaker = CircuitBreaker(threshold)
        self._breakers[handler] = breaker

        def wrapped(data: Any, h: Callable[[Any], None] = handler, b: CircuitBreaker = breaker) -> None:
            try:
                b.call(h, data)
            except Exception:
                # suppress to allow other handlers to process
                pass

        self.dispatcher.subscribe(event_type, wrapped)

    def publish(self, event_type: str, data: Any) -> None:
        """Publish an event after basic validation."""
        if not isinstance(event_type, str):
            raise TypeError("event_type must be str")
        self.dispatcher.dispatch(event_type, data)


_dispatcher = EventDispatcher()
_router = MessageRouter(_dispatcher)


def subscribe_to_brain_region(
    region: str, event_type: str, handler: Callable[[Any], None], threshold: int = 3
) -> None:
    """Public API to subscribe a brain region to an event type."""
    _router.register(region, event_type, handler, threshold)


def publish_neural_event(event_type: str, data: Any) -> None:
    """Public API to publish neural events to the bus."""
    _router.publish(event_type, data)
