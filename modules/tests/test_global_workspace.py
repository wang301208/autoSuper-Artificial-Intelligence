import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "backend/monitoring")))

from global_workspace import GlobalWorkspace


class DummyModule:
    def __init__(self) -> None:
        self.received = []

    def receive_broadcast(self, sender, state, attention):
        self.received.append((sender, state, attention))


def test_broadcast_propagates_state() -> None:
    gw = GlobalWorkspace()
    a = DummyModule()
    b = DummyModule()
    gw.register_module("a", a)
    gw.register_module("b", b)
    gw.broadcast("a", {"value": 1}, 0.5)
    assert b.received == [("a", {"value": 1}, 0.5)]
    assert gw.state("a") == {"value": 1}
    assert gw.attention("a") == 0.5


def test_subscribe_state_receives_updates() -> None:
    gw = GlobalWorkspace()
    received = []
    gw.subscribe_state("self_model", lambda s: received.append(s))
    gw.broadcast("self_model", {"agent": "a", "summary": "hi"})
    assert received == [{"agent": "a", "summary": "hi"}]
