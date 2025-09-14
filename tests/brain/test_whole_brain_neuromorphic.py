import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain import WholeBrainSimulation


def test_whole_brain_neuromorphic_cycle():
    brain = WholeBrainSimulation(neuromorphic=True)
    input_data = {
        "image": [0.9, 0.1],
        "sound": [0.5],
        "touch": [0.3, 0.7],
        "text": "good",
        "is_salient": True,
    }
    action = brain.process_cycle(input_data)
    assert "executed" in action
    assert brain.last_perception["vision"]["spike_counts"] == [1, 1]
    assert brain.last_perception["audio"]["spike_counts"] == [1]
    assert brain.last_perception["touch"]["spike_counts"] == [1, 1]
