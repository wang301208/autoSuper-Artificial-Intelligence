import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.security import NeuralSecurityGuard


def test_validate_neural_input_handles_threats():
    guard = NeuralSecurityGuard()
    data = {"values": [0.2, 99.0], "text": "hello trigger"}
    cleaned = guard.validate_neural_input(data)
    assert max(cleaned["values"]) <= guard.adversarial_detector.threshold
    assert "trigger" not in cleaned["text"]


def test_memory_checker_repairs_corruption():
    guard = NeuralSecurityGuard()
    memory = {"status": "ok"}
    assert guard.protect_neural_memory(memory) == []
    memory["status"] = "tampered"
    issues = guard.protect_neural_memory(memory)
    assert issues == ["status"]
    assert memory["status"] == "ok"
