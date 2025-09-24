import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain import WholeBrainSimulation


def test_whole_brain_neuromorphic_cycle():
    brain = WholeBrainSimulation(neuromorphic=True)
    vision_stream = iter([[0.9, 0.1], [0.4, 0.6]])
    input_data = {
        "streams": {"vision": vision_stream},
        "sound": [0.5],
        "touch": [0.3, 0.7],
        "text": "good",
        "is_salient": True,
        "context": {"task": "explore"},
    }
    result = brain.process_cycle(input_data)
    assert result.metadata["executed_action"].startswith("executed")
    assert result.intent.plan
    assert result.metadata["policy"] == "heuristic"
    assert result.metadata["policy_metadata"]["confidence_calibrated"] is True
    assert result.metrics.get("cycle_index", 0.0) >= 1
    assert result.energy_used > 0
    assert "curiosity_drive" in result.metrics
    assert result.emotion.mood <= 1.0
    assert len(brain.perception_history) == 1
    assert "vision" in brain.last_perception.modalities
    assert brain.last_context.get("task") == "explore"
    assert brain.telemetry_log[-1]["modalities"]["vision"] in {"stream", "cached"}
    follow_up = {
        "streams": {"vision": vision_stream},
        "sound": [0.2],
        "touch": [0.6, 0.4],
        "context": {"task": "focus"},
    }
    second_result = brain.process_cycle(follow_up)
    assert len(brain.perception_history) == 2
    assert len(brain.decision_history) == 2
    assert brain.telemetry_log[-1]["cycle_index"] == brain.cycle_index
    assert second_result.intent.confidence <= 1.0
    assert brain.last_context.get("task") == "focus"
    assert brain.telemetry_log[-1]["cognitive_plan"]
    modulation = brain.get_strategy_modulation()
    assert modulation["curiosity_drive"] == brain.curiosity.drive


def test_energy_usage_varies_with_activity():
    brain = WholeBrainSimulation(neuromorphic=True)
    active = {"image": [1.0, 0.6], "sound": [0.9], "touch": [0.2, 0.8]}
    idle = {"image": [0.0, 0.0], "sound": [0.0], "touch": [0.0, 0.0]}
    energy_active = brain.process_cycle(active).energy_used
    idle_result = brain.process_cycle(idle)
    energy_idle = idle_result.energy_used
    assert energy_active > energy_idle
    assert idle_result.idle_skipped > 0
