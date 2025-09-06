import os
import sys

import pytest

# Ensure repository root is on PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.self_model import SelfModel
from backend.memory.long_term import LongTermMemory
class DummyBus:
    def __init__(self) -> None:
        self._subs = []

    def subscribe(self, _topic, handler):
        self._subs.append(handler)

    def publish(self, _topic, event):
        for h in self._subs:
            h(event)


def test_assess_state_and_event():
    data = {"cpu": 1.0, "memory": 2.0}
    env_pred = {"avg_cpu": 2.0, "avg_memory": 3.0}
    self_model = SelfModel(LongTermMemory(":memory:"))

    metrics, summary = self_model.assess_state(data, env_pred, "idle")

    assert metrics["cpu"] == pytest.approx(0.8)
    assert metrics["memory"] == pytest.approx(1.7)
    assert summary in self_model.history

    bus = DummyBus()
    received = []
    bus.subscribe("agent.self_awareness", lambda e: received.append(e))
    bus.publish("agent.self_awareness", {"agent": "test", "summary": summary})

    assert received and received[0]["summary"] == summary
