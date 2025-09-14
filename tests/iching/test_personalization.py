import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from modules.iching import PersonalizedHexagramEngine


def test_personalized_interpretation_varies():
    engine = PersonalizedHexagramEngine()

    alice = engine.get_profile("alice")
    alice.set_preference("element", "water")
    alice.set_trait("risk_tolerance", "low")

    bob = engine.get_profile("bob")
    bob.set_preference("element", "fire")
    bob.set_trait("risk_tolerance", "high")

    result_alice = engine.interpret("alice", "01")
    result_bob = engine.interpret("bob", "01")

    assert "Flow around obstacles" in result_alice
    assert "Take cautious steps" in result_alice

    assert "Pursue your goals with passion" in result_bob
    assert "Bold moves" in result_bob

    assert alice.history == ["01"]
    assert bob.history == ["01"]
