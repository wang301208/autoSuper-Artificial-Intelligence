import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from modules.brain.self_learning import SelfLearningBrain


def test_curiosity_driven_learning_improves_prediction():
    brain = SelfLearningBrain()
    sample = {
        "state": "s1",
        "agent_id": "agent",
        "usage": {"cpu": 1.0, "memory": 2.0},
        "reward": 1.0,
    }

    before = brain.world_model.predict("agent")
    err_before = abs(before["cpu"] - 1.0) + abs(before["memory"] - 2.0)

    brain.curiosity_driven_learning(sample)

    after = brain.world_model.predict("agent")
    err_after = abs(after["cpu"] - 1.0) + abs(after["memory"] - 2.0)

    assert "s1" in brain.memory
    assert err_after < err_before
