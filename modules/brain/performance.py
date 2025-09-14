"""Brain performance profiling and optimization utilities.

This module provides simple profilers for spike patterns, energy
consumption, and latency. It also offers helper functions to collect
metrics across these profilers and automatically apply basic optimisation
strategies based on the collected data.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List, Sequence, Tuple


@dataclass
class SpikePatternAnalyzer:
    """Analyse neural spike trains.

    The implementation is intentionally lightweight: given a sequence of
    spike timestamps (in seconds) it computes the average firing rate.
    """

    def analyse(self, spikes: Sequence[float]) -> float:
        """Return average firing rate of the provided spike times."""
        if len(spikes) < 2:
            return 0.0
        duration = spikes[-1] - spikes[0]
        if duration <= 0:
            return 0.0
        return (len(spikes) - 1) / duration


@dataclass
class EnergyConsumptionProfiler:
    """Profile energy usage of a brain region or algorithm."""

    def profile(self, samples: Sequence[float]) -> Dict[str, float]:
        total = float(sum(samples))
        avg = total / len(samples) if samples else 0.0
        return {"total": total, "average": avg}


@dataclass
class LatencyProfiler:
    """Measure latency statistics for operations."""

    def profile(self, latencies: Sequence[float]) -> Dict[str, float]:
        if not latencies:
            return {"average": 0.0, "max": 0.0}
        return {"average": mean(latencies), "max": max(latencies)}


def profile_brain_performance(
    spikes: Sequence[float],
    energy_samples: Sequence[float],
    latencies: Sequence[float],
) -> Dict[str, Any]:
    """Collect performance metrics from the monitoring components.

    Parameters
    ----------
    spikes: Sequence[float]
        Spike timestamps in seconds.
    energy_samples: Sequence[float]
        Energy consumption samples in arbitrary units.
    latencies: Sequence[float]
        Operation latencies in seconds.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing raw metrics and optimisation suggestions.
    """

    spike_analyser = SpikePatternAnalyzer()
    energy_profiler = EnergyConsumptionProfiler()
    latency_profiler = LatencyProfiler()

    metrics = {
        "spike_rate": spike_analyser.analyse(spikes),
        "energy": energy_profiler.profile(energy_samples),
        "latency": latency_profiler.profile(latencies),
    }

    suggestions: List[str] = []
    if metrics["energy"]["average"] > 50:
        suggestions.append("Reduce energy consumption: consider lower power mode")
    if metrics["latency"]["average"] > 0.1:
        suggestions.append("High latency detected: optimise processing pipeline")
    if metrics["spike_rate"] > 100:
        suggestions.append("Spike rate unusually high: adjust stimulation levels")

    return {"metrics": metrics, "suggestions": suggestions}


def auto_optimize_performance(
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    thresholds: Dict[str, float] | None = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Adjust configuration parameters based on collected metrics.

    Parameters
    ----------
    metrics
        Output from :func:`profile_brain_performance`.
    config
        Configuration dictionary to be adjusted.
    thresholds
        Optional thresholds overriding the defaults.

    Returns
    -------
    Tuple[Dict[str, Any], List[str]]
        The updated configuration and a list of applied optimisation
        actions.
    """

    thresholds = thresholds or {}
    actions: List[str] = []

    avg_energy = metrics.get("energy", {}).get("average", 0.0)
    if avg_energy > thresholds.get("energy", 50.0):
        config["power_mode"] = "low"
        actions.append("power_mode -> low")

    avg_latency = metrics.get("latency", {}).get("average", 0.0)
    if avg_latency > thresholds.get("latency", 0.1):
        config["async"] = True
        actions.append("async processing enabled")

    spike_rate = metrics.get("spike_rate", 0.0)
    if spike_rate > thresholds.get("spike_rate", 100.0):
        config["stimulation_level"] = config.get("stimulation_level", 1.0) * 0.9
        actions.append("stimulation level reduced")

    return config, actions
