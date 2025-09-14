import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from modules.brain.message_bus import publish_neural_event, subscribe_to_brain_region


def test_bus_routes_messages():
    received = []

    def handler(data):
        received.append(data)

    subscribe_to_brain_region("motor", "move", handler)
    publish_neural_event("move", {"direction": "left"})
    assert received == [{"direction": "left"}]


def test_circuit_breaker_isolates_failures():
    received = []
    fail_count = {"count": 0}

    def good_handler(data):
        received.append(data)

    def failing_handler(data):
        fail_count["count"] += 1
        raise ValueError("boom")

    subscribe_to_brain_region("good", "signal", good_handler)
    subscribe_to_brain_region("bad", "signal", failing_handler)

    for _ in range(4):
        publish_neural_event("signal", 1)

    assert received == [1, 1, 1, 1]
    # circuit breaker stops calling after 3 failures
    assert fail_count["count"] == 3
