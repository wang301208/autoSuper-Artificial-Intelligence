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
    result = brain.process_cycle(input_data)
    assert "executed" in result["action"]
    assert result["energy_used"] > 0
    assert brain.last_perception["vision"]["spike_counts"] == [1, 0]
    assert brain.last_perception["audio"]["spike_counts"] == [1]
    assert brain.last_perception["touch"]["spike_counts"] == [0, 1]


def test_energy_usage_varies_with_activity():
    brain = WholeBrainSimulation(neuromorphic=True)
    active = {"image": [1.0, 0.6], "sound": [0.9], "touch": [0.2, 0.8]}
    idle = {"image": [0.0, 0.0], "sound": [0.0], "touch": [0.0, 0.0]}
    energy_active = brain.process_cycle(active)["energy_used"]
    idle_result = brain.process_cycle(idle)
    energy_idle = idle_result["energy_used"]
    assert energy_active > energy_idle
    assert idle_result["idle_skipped"] > 0
