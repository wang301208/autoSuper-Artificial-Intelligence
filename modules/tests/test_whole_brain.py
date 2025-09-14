import os
import sys

sys.path.insert(0, os.path.abspath(os.getcwd()))

from modules.brain.whole_brain import WholeBrainSimulation


def test_process_cycle_returns_action_and_broadcast():
    brain = WholeBrainSimulation()
    input_data = {
        "image": [1.0],
        "sound": [1.0],
        "touch": [1.0],
        "text": "good",  # positive to trigger happy emotion
        "is_salient": True,
    }

    result = brain.process_cycle(input_data)

    assert result["action"].startswith("executed")
    assert result["energy_used"] >= 0
    assert brain.consciousness.workspace.broadcasts

