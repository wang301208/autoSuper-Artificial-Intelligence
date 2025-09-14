import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.getcwd()))

from modules.brain.performance import (
    EnergyConsumptionProfiler,
    LatencyProfiler,
    auto_optimize_performance,
    profile_brain_performance,
)


def test_energy_consumption_profiler_accuracy():
    profiler = EnergyConsumptionProfiler()
    data = [2.0, 3.0, 5.0]
    result = profiler.profile(data)
    assert result["total"] == 10.0
    assert result["average"] == pytest.approx(10.0 / 3)


def test_latency_profiler_accuracy():
    profiler = LatencyProfiler()
    latencies = [0.1, 0.2, 0.05]
    result = profiler.profile(latencies)
    assert result["average"] == pytest.approx(sum(latencies) / len(latencies))
    assert result["max"] == 0.2


def test_profile_and_auto_optimization():
    spikes = [0.0, 0.01, 0.015, 0.02]
    energy = [60.0, 80.0]
    latencies = [0.2, 0.15]
    outcome = profile_brain_performance(spikes, energy, latencies)
    assert outcome["suggestions"]

    config = {}
    updated, actions = auto_optimize_performance(outcome["metrics"], config)
    assert updated["power_mode"] == "low"
    assert updated["async"] is True
    assert "stimulation level reduced" in actions
