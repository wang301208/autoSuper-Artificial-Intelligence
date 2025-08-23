"""Event bus abstraction for AutoGPT.

This module exposes a global ``event_bus`` instance that other modules can use to
publish or subscribe to events. The backend is selected via the ``EVENT_BUS_BACKEND``
environment variable and supports ``kafka`` and ``nats`` when the relevant client
libraries are available. If no backend is configured or the required libraries are
missing, an in-process pub/sub implementation is used as a fallback.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Callable, Dict, List

__all__ = ["event_bus", "publish", "subscribe"]


class InProcessEventBus:
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

    def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        with self._lock:
            self._subscribers.setdefault(topic, []).append(handler)


class KafkaEventBus:
    """Kafka based event bus using :mod:`kafka-python`.

    This is a minimal implementation intended mainly for publishing events. It will
    fall back to :class:`InProcessEventBus` if the ``kafka`` library is not available
    or a connection cannot be established.
    """

    def __init__(self, brokers: str) -> None:
        try:
            from kafka import KafkaProducer, KafkaConsumer  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("kafka-python is required for KafkaEventBus") from exc

        self._producer = KafkaProducer(
            bootstrap_servers=brokers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        self._brokers = brokers

    def publish(self, topic: str, event: Dict[str, Any]) -> None:  # pragma: no cover - network
        self._producer.send(topic, event)

    def subscribe(
        self, topic: str, handler: Callable[[Dict[str, Any]], None]
    ) -> None:  # pragma: no cover - network
        from kafka import KafkaConsumer  # type: ignore

        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=self._brokers,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="earliest",
        )

        def _consume() -> None:
            for msg in consumer:
                handler(msg.value)

        threading.Thread(target=_consume, daemon=True).start()


class NATSEventBus:
    """NATS based event bus using :mod:`nats-py`.

    Only basic publish/subscribe functionality is provided and it is intended for
    simple use cases. The class is only instantiated when the ``nats-py`` library is
    installed and a connection can be made; otherwise the fallback bus is used.
    """

    def __init__(self, servers: str) -> None:
        import asyncio

        try:
            from nats.aio.client import Client as NATS  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("nats-py is required for NATSEventBus") from exc

        self._loop = asyncio.get_event_loop()
        self._nats = NATS()
        self._loop.run_until_complete(self._nats.connect(servers=servers))

    def publish(self, topic: str, event: Dict[str, Any]) -> None:  # pragma: no cover - network
        payload = json.dumps(event).encode("utf-8")
        self._loop.create_task(self._nats.publish(topic, payload))

    def subscribe(
        self, topic: str, handler: Callable[[Dict[str, Any]], None]
    ) -> None:  # pragma: no cover - network
        async def cb(msg):
            handler(json.loads(msg.data.decode("utf-8")))

        self._loop.create_task(self._nats.subscribe(topic, cb=cb))


def _create_event_bus() -> InProcessEventBus:
    backend = os.getenv("EVENT_BUS_BACKEND", "inproc").lower()
    if backend == "kafka":
        brokers = os.getenv("EVENT_BUS_BROKERS", "localhost:9092")
        try:
            return KafkaEventBus(brokers)
        except Exception:
            pass
    elif backend == "nats":
        servers = os.getenv("EVENT_BUS_SERVERS", "nats://127.0.0.1:4222")
        try:
            return NATSEventBus(servers)
        except Exception:
            pass
    return InProcessEventBus()


event_bus = _create_event_bus()


def publish(topic: str, event: Dict[str, Any]) -> None:
    """Publish *event* on *topic* using the configured event bus."""

    event_bus.publish(topic, event)


def subscribe(topic: str, handler: Callable[[Dict[str, Any]], None]) -> None:
    """Subscribe *handler* to events published on *topic*."""

    event_bus.subscribe(topic, handler)
