"""Utilities for streaming data ingestion using an event queue.

The :class:`StreamingDataIngestor` subscribes to an event bus topic and
buffers incoming events in a local queue. A consumer can periodically drain
this queue and feed the resulting samples into an online learning routine.

The ingestor optionally accepts an active learning sampler which can be used
to prioritise which samples from the queue are returned.
"""

from __future__ import annotations

from queue import Empty, Queue
from typing import Any, Callable, Dict, List

from events import EventBus, create_event_bus
from backend.ml.active_sampler import ActiveLearningSampler


class StreamingDataIngestor:
    """Subscribe to events and expose them as a stream of training samples."""

    def __init__(
        self,
        bus: EventBus | None = None,
        *,
        topic: str = "training.sample",
        sampler: ActiveLearningSampler | None = None,
    ) -> None:
        self.bus = bus or create_event_bus("inmemory")
        self.topic = topic
        self.queue: Queue[Dict[str, Any]] = Queue()
        self.bus.subscribe(topic, self.queue.put)
        self.sampler = sampler

    def drain(self) -> List[Dict[str, Any]]:
        """Return all queued events, optionally prioritised by the sampler."""
        batch: List[Dict[str, Any]] = []
        while True:
            try:
                batch.append(self.queue.get_nowait())
            except Empty:
                break
        if self.sampler and batch:
            features = [e.get("features") for e in batch]
            k = len(batch)
            idx = self.sampler.select(features=features, k=k)
            batch = [batch[i] for i in idx]
        return batch

    def stream(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Process all queued events using ``handler``."""
        for event in self.drain():
            handler(event)
