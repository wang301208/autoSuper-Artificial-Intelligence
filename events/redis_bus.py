"""Redis based event bus implementation."""

from __future__ import annotations

import json
import threading
from typing import Any, Callable, Dict

import redis

from . import EventBus


class RedisEventBus(EventBus):
    """Event bus using Redis Pub/Sub."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: str | None = None,
        db: int = 0,
    ) -> None:
        self._redis = redis.Redis(host=host, port=port, password=password, db=db)

    def publish(self, topic: str, event: Dict[str, Any]) -> None:
        self._redis.publish(topic, json.dumps(event))

    def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        pubsub = self._redis.pubsub()
        pubsub.subscribe(topic)

        def _listen() -> None:
            for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                try:
                    data = json.loads(message["data"])
                except Exception:
                    continue
                try:
                    handler(data)
                except Exception:
                    pass

        thread = threading.Thread(target=_listen, daemon=True)
        thread.start()


__all__ = ["RedisEventBus"]

