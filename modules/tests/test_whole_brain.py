import os
import sys

sys.path.insert(0, os.path.abspath(os.getcwd()))

from modules.brain.whole_brain import WholeBrainSimulation


def test_process_cycle_returns_action_and_broadcast():
    brain = WholeBrainSimulation()
    input_data = {
        "image": "img",
        "sound": "snd",
        "touch": "tch",
        "text": "good",  # positive to trigger happy emotion
        "is_salient": True,
    }

    action = brain.process_cycle(input_data)

    assert action.startswith("executed")
    assert brain.consciousness.workspace.broadcasts

