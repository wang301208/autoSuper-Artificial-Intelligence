import os
import sys


sys.path.insert(0, os.path.abspath(os.getcwd()))

from modules.brain.whole_brain import WholeBrainSimulation
from modules.brain.state import BrainCycleResult, BrainRuntimeConfig


def test_process_cycle_returns_action_and_broadcast():
    brain = WholeBrainSimulation()
    input_data = {
        "image": [1.0],
        "sound": [1.0],
        "touch": [1.0],
        "text": "good",
        "is_salient": True,
        "context": {"task": "greet", "safety": 0.5},
    }

    result = brain.process_cycle(input_data)

    assert isinstance(result, BrainCycleResult)
    assert result.metadata["executed_action"].startswith("executed")
    assert result.energy_used >= 0
    assert "curiosity_drive" in result.metrics
    assert "plan_length" in result.metrics
    assert "strategy_bias_approach" in result.metrics
    assert result.intent.plan
    assert 0.0 <= result.intent.confidence <= 1.0
    assert result.metadata["context_task"] == "greet"
    assert brain.get_decision_trace()


def test_process_cycle_handles_nested_signals():
    brain = WholeBrainSimulation()
    input_data = {
        "image": [[0.1, 0.2], [0.3, 0.4]],
        "sound": [[0.5, 0.6, 0.7]],
        "text": "neutral",
    }

    result = brain.process_cycle(input_data)

    vision = brain.last_perception.modalities.get("vision")
    assert result.intent.intention
    assert result.energy_used >= 0
    assert vision is not None
    assert len(vision["spike_counts"]) <= brain.max_neurons
    assert "novelty_signal" in result.metrics


def test_process_cycle_logs_invalid_signal(caplog):
    brain = WholeBrainSimulation()
    with caplog.at_level("DEBUG", logger="modules.brain.whole_brain"):
        brain.process_cycle({"image": {"bad": "data"}, "text": ""})
    assert "Unsupported sensory signal type" in caplog.text


def test_spiking_cache_respects_limit():
    brain = WholeBrainSimulation()
    brain.max_cache_size = 2
    signals = [[1.0], [1.0, 1.0], [1.0, 1.0, 1.0], [1.0] * 8]
    for signal in signals:
        brain.process_cycle({"image": signal})
    assert len(brain._spiking_cache) <= brain.max_cache_size


def test_process_cycle_latency_encoding():
    brain = WholeBrainSimulation()
    brain.neuromorphic_encoding = "latency"
    brain.encoding_time_scale = 0.5
    result = brain.process_cycle({"image": [0.2, 0.8], "text": ""})
    assert result.energy_used >= 0
    vision = brain.last_perception.modalities.get("vision")
    assert vision is not None
    assert len(vision["spike_counts"]) <= brain.max_neurons
    assert result.metrics.get("cycle_index", 0.0) >= 1.0


def test_update_config_disables_metrics():
    brain = WholeBrainSimulation()
    config = BrainRuntimeConfig(metrics_enabled=False, enable_self_learning=False)
    brain.update_config(config)
    result = brain.process_cycle({"text": ""})
    assert result.metrics == {}
