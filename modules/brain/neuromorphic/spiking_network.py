from __future__ import annotations

import numpy as np
import argparse
import asyncio
import concurrent.futures
import heapq
import json
import math
import random
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import Executor
from typing import Any, Callable, Dict, Mapping, Sequence, List, Optional

from modules.brain.neuroplasticity import Neuroplasticity
from .temporal_encoding import decode_average_rate, decode_spike_counts, latency_encode, rate_encode


@dataclass
class SpikingNetworkConfig:
    """Configuration container for building spiking neural networks."""

    n_neurons: int
    neuron: str = "lif"
    neuron_params: Mapping[str, Any] = field(default_factory=dict)
    weights: Sequence[Sequence[float]] | None = None
    idle_skip: bool = False
    plasticity: str | None = "stdp"
    learning_rate: float = 0.1
    max_duration: int | None = None
    convergence_window: int | None = None
    convergence_threshold: float | None = None
    convergence_patience: int = 3

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SpikingNetworkConfig":
        return cls(
            n_neurons=data["n_neurons"],
            neuron=data.get("neuron", "lif"),
            neuron_params=data.get("neuron_params", {}),
            weights=data.get("weights"),
            idle_skip=data.get("idle_skip", False),
            plasticity=data.get("plasticity", "stdp"),
            learning_rate=data.get("learning_rate", 0.1),
            max_duration=data.get("max_duration"),
            convergence_window=data.get("convergence_window"),
            convergence_threshold=data.get("convergence_threshold"),
            convergence_patience=data.get("convergence_patience", 3),
        )

    def create(self) -> "SpikingNeuralNetwork":
        neuron_map = {
            "lif": LIFNeuronModel,
            "adex": AdExNeuronModel,
        }
        neuron_key = self.neuron.lower()
        if neuron_key not in neuron_map:
            raise ValueError(f"Unknown neuron model '{self.neuron}'")
        neuron_cls = neuron_map[neuron_key]
        return SpikingNeuralNetwork(
            self.n_neurons,
            weights=self.weights,
            idle_skip=self.idle_skip,
            neuron_model_cls=neuron_cls,
            neuron_model_kwargs=dict(self.neuron_params),
            plasticity_mode=self.plasticity,
            learning_rate=self.learning_rate,
            max_duration=self.max_duration,
            convergence_window=self.convergence_window,
            convergence_threshold=self.convergence_threshold,
            convergence_patience=self.convergence_patience,
        )

    def create_backend(self, **backend_kwargs) -> "NeuromorphicBackend":
        """Build a reusable backend wrapping the configured spiking network."""

        return NeuromorphicBackend(config=self, **backend_kwargs)



class EventQueue:
    """Priority queue managing spike events by timestamp."""

    def __init__(self) -> None:
        self._queue: list[tuple[float, list[float]]] = []

    def push(self, time: float, inputs: list[float]) -> None:
        heapq.heappush(self._queue, (time, inputs))

    def pop(self) -> tuple[float, list[float]]:
        return heapq.heappop(self._queue)

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return bool(self._queue)


class NeuronModel(ABC):
    """Abstract base class for neuron dynamics."""

    size: int

    @abstractmethod
    def step(self, inputs: Sequence[float]) -> list[int]:
        """Advance the neuron state given input currents."""

    @abstractmethod
    def reset_state(self) -> None:
        """Reset membrane potentials and auxiliary state."""


class LIFNeuronModel(NeuronModel):
    """Leaky integrate-and-fire neuron population using NumPy."""

    def __init__(
        self,
        size: int,
        *,
        decay: float = 0.9,
        threshold: float = 1.0,
        reset: float = 0.0,
        refractory_period: int = 0,
        dynamic_threshold: float = 0.0,
        noise: float | None = None,
    ) -> None:
        self.size = size
        self.decay = decay
        self.threshold = threshold
        self.reset_value = reset
        self.refractory_period = refractory_period
        self.dynamic_threshold = dynamic_threshold
        self.noise = noise
        self.potentials = np.zeros(size, dtype=float)
        self.refractory = np.zeros(size, dtype=int)
        self.adaptation = np.zeros(size, dtype=float)

    def reset_state(self) -> None:
        self.potentials.fill(self.reset_value)
        self.refractory.fill(0)
        self.adaptation.fill(0.0)

    def step(self, inputs: Sequence[float]) -> list[int]:
        current = np.asarray(inputs, dtype=float)
        if current.shape[0] != self.size:
            raise ValueError("input size does not match neuron population")

        active = self.refractory <= 0
        inactive = ~active

        self.refractory[inactive] -= 1
        self.refractory = np.maximum(self.refractory, 0)

        self.potentials[active] = self.potentials[active] * self.decay + current[active]
        self.potentials[inactive] = self.reset_value

        if self.noise is not None:
            noise = np.random.normal(0.0, self.noise, size=self.size)
            self.potentials[active] += noise[active]

        thresholds = self.threshold + self.adaptation
        fire_mask = active & (self.potentials >= thresholds)

        spikes = fire_mask.astype(int)
        self.potentials[fire_mask] = self.reset_value
        self.refractory[fire_mask] = self.refractory_period
        self.adaptation[fire_mask] += self.dynamic_threshold
        self.adaptation[~fire_mask] *= self.decay
        return spikes.tolist()


class AdExNeuronModel(NeuronModel):
    """Adaptive exponential integrate-and-fire neurons (NumPy)."""

    def __init__(
        self,
        size: int,
        *,
        tau_m: float = 20.0,
        tau_w: float = 100.0,
        a: float = 0.0,
        b: float = 0.02,
        v_reset: float = -65.0,
        v_threshold: float = -50.0,
        delta_t: float = 2.0,
        v_peak: float = 20.0,
        timestep: float = 1.0,
    ) -> None:
        self.size = size
        self.tau_m = tau_m
        self.tau_w = tau_w
        self.a = a
        self.b = b
        self.v_reset = v_reset
        self.v_threshold = v_threshold
        self.delta_t = delta_t
        self.v_peak = v_peak
        self.timestep = timestep
        self.v = np.full(size, v_reset, dtype=float)
        self.w = np.zeros(size, dtype=float)

    def reset_state(self) -> None:
        self.v.fill(self.v_reset)
        self.w.fill(0.0)

    def step(self, inputs: Sequence[float]) -> list[int]:
        current = np.asarray(inputs, dtype=float)
        if current.shape[0] != self.size:
            raise ValueError("input size does not match neuron population")

        dv = (
            -(self.v - self.v_reset)
            + self.delta_t * np.exp((self.v - self.v_threshold) / self.delta_t)
            - self.w
            + current
        ) * (self.timestep / self.tau_m)
        self.v += dv
        dw = (self.a * (self.v - self.v_reset) - self.w) * (self.timestep / self.tau_w)
        self.w += dw

        fire_mask = self.v >= self.v_peak
        spikes = fire_mask.astype(int)
        self.v[fire_mask] = self.v_reset
        self.w[fire_mask] += self.b
        return spikes.tolist()


class SynapseModel(ABC):
    """Abstract base class for synaptic connectivity."""

    @abstractmethod
    def propagate(self, pre_spikes: Sequence[int]) -> list[float]:
        """Propagate spikes to produce postsynaptic currents."""

    @abstractmethod
    def adapt(
        self,
        pre_spike_times: Sequence[float | None],
        post_spike_times: Sequence[float | None],
    ) -> None:
        """Update weights given spike timing."""

    @abstractmethod
    def reset_state(self) -> None:
        """Reset synaptic weights and plasticity state."""


class DenseSynapseModel(SynapseModel):
    """Fully connected synapses with optional plasticity using NumPy."""

    def __init__(
        self,
        weights: Sequence[Sequence[float]],
        *,
        learning_rate: float = 0.1,
        plasticity: Neuroplasticity | None = None,
    ) -> None:
        self._initial_weights = np.asarray(weights, dtype=float)
        self.weights = self._initial_weights.copy()
        self.base_learning_rate = float(learning_rate)
        self.learning_rate = float(learning_rate)
        self._plasticity_cls = plasticity.__class__ if plasticity is not None else None
        self.plasticity = plasticity if plasticity is not None else Neuroplasticity()
        self.weight_decay = 0.0
        self._modulation_cache: Dict[str, float] = {}

    def reset_state(self) -> None:
        self.weights = self._initial_weights.copy()
        if self._plasticity_cls is not None:
            self.plasticity = self._plasticity_cls()
        self.learning_rate = self.base_learning_rate
        self.weight_decay = 0.0
        self._modulation_cache = {}
        if hasattr(self.plasticity, "update_modulation"):
            self.plasticity.update_modulation(None)

    def propagate(self, pre_spikes: Sequence[int]) -> list[float]:
        pre = np.asarray(pre_spikes, dtype=float)
        postsynaptic = pre @ self.weights
        return postsynaptic.tolist()

    def adapt(
        self,
        pre_spike_times: Sequence[float | None],
        post_spike_times: Sequence[float | None],
    ) -> None:
        if self.plasticity is None:
            return
        pre_times = [t for t in enumerate(pre_spike_times) if t[1] is not None]
        post_times = [t for t in enumerate(post_spike_times) if t[1] is not None]
        for pre_idx, pre_time in pre_times:
            for post_idx, post_time in post_times:
                delta = self.plasticity.adapt_connections(pre_time, post_time)
                self.weights[pre_idx, post_idx] += self.learning_rate * delta
        if self.weight_decay:
            self.weights *= (1.0 - self.weight_decay)

    def update_modulation(self, modulation: Mapping[str, float] | None) -> None:
        if not modulation:
            self.learning_rate = self.base_learning_rate
            self.weight_decay = 0.0
            self._modulation_cache = {}
            if hasattr(self.plasticity, "update_modulation"):
                self.plasticity.update_modulation(None)
            return
        filtered: Dict[str, float] = {
            key: float(value)
            for key, value in modulation.items()
            if isinstance(value, (int, float))
        }
        amplitude = float(np.clip(filtered.get("amplitude_norm", filtered.get("amplitude", 0.0)), 0.0, 1.0))
        synchrony = float(np.clip(filtered.get("synchrony_norm", filtered.get("synchrony_index", 0.0)), 0.0, 1.0))
        rhythmicity = float(np.clip(filtered.get("rhythmicity", 0.0), 0.0, 1.0))
        gate = float(np.clip(filtered.get("plasticity_gate", (amplitude + synchrony) * 0.5), 0.0, 2.0))
        learning_gain = 0.5 + amplitude * 0.75 + synchrony * 0.25
        learning_gain += rhythmicity * 0.25
        self.learning_rate = self.base_learning_rate * max(0.1, learning_gain)
        decay_term = (1.0 - synchrony) * 0.05 + max(0.0, 0.5 - amplitude) * 0.02
        self.weight_decay = float(np.clip(decay_term, 0.0, 0.2))
        self._modulation_cache = {
            "amplitude": amplitude,
            "synchrony": synchrony,
            "rhythmicity": rhythmicity,
            "plasticity_gate": gate,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }
        if hasattr(self.plasticity, "update_modulation"):
            self.plasticity.update_modulation({
                **filtered,
                "plasticity_gate": gate,
                "learning_rate": self.learning_rate,
            })

    @property
    def modulation_cache(self) -> Dict[str, float]:
        return dict(self._modulation_cache)


class SpikingNeuralNetwork:
    """Spiking neural network with pluggable neuron and synapse models."""

    LeakyIntegrateFireNeurons = LIFNeuronModel
    AdaptiveExponentialNeurons = AdExNeuronModel
    DynamicSynapses = DenseSynapseModel

    def __init__(
        self,
        n_neurons,
        *,
        decay=0.9,
        threshold=1.0,
        reset=0.0,
        weights=None,
        refractory_period=0,
        dynamic_threshold=0.0,
        noise=None,
        idle_skip=False,
        neuron_model: NeuronModel | None = None,
        neuron_model_cls: type[NeuronModel] | None = None,
        neuron_model_kwargs: dict | None = None,
        synapse_model: SynapseModel | None = None,
        plasticity_mode: str | None = "stdp",
        learning_rate: float = 0.1,
        max_duration: int | None = None,
        convergence_window: int | None = None,
        convergence_threshold: float | None = None,
        convergence_patience: int | None = 3,
    ) -> None:
        if neuron_model is None:
            if neuron_model_cls is None:
                neuron_model = LIFNeuronModel(
                    n_neurons,
                    decay=decay,
                    threshold=threshold,
                    reset=reset,
                    refractory_period=refractory_period,
                    dynamic_threshold=dynamic_threshold,
                    noise=noise,
                )
            else:
                params = neuron_model_kwargs or {}
                neuron_model = neuron_model_cls(n_neurons, **params)
        elif neuron_model.size != n_neurons:
            raise ValueError("neuron_model size must match n_neurons")

        if synapse_model is None:
            if weights is None:
                weights = np.eye(n_neurons, dtype=float)
            plasticity = None
            if plasticity_mode is None:
                plasticity = None
            elif plasticity_mode.lower() == "stdp":
                plasticity = Neuroplasticity()
            synapse_model = DenseSynapseModel(
                weights,
                learning_rate=learning_rate,
                plasticity=plasticity,
            )

        self.neurons = neuron_model
        self.synapses = synapse_model
        self.spike_times = [None] * n_neurons
        self._modulation_state: Dict[str, float] = {}
        if hasattr(self.synapses, "update_modulation"):
            self.synapses.update_modulation(None)
        self.idle_skip = idle_skip
        self.energy_usage = 0
        self.idle_skipped_cycles = 0
        self.max_duration = (
            int(max_duration) if max_duration is not None else None
        )
        if self.max_duration is not None and self.max_duration <= 0:
            self.max_duration = 1
        self.convergence_threshold = (
            float(convergence_threshold)
            if convergence_threshold is not None
            else None
        )
        if convergence_window is not None:
            window_value = int(convergence_window)
            self.convergence_window = window_value if window_value > 0 else None
        else:
            self.convergence_window = None
        if convergence_patience is None:
            patience_value = 3
        else:
            patience_value = int(convergence_patience)
        self.convergence_patience = patience_value if patience_value > 0 else 1

    def reset_state(self) -> None:
        if hasattr(self.neurons, "reset_state"):
            self.neurons.reset_state()
        if hasattr(self.synapses, "reset_state"):
            self.synapses.reset_state()
            if hasattr(self.synapses, "update_modulation"):
                if self._modulation_state:
                    self.synapses.update_modulation(self._modulation_state)
                else:
                    self.synapses.update_modulation(None)
        self.spike_times = [None] * len(self.spike_times)
        self.energy_usage = 0
        self.idle_skipped_cycles = 0

    def apply_modulation(self, modulation: Mapping[str, float] | None) -> None:
        if modulation:
            state = {
                key: float(value)
                for key, value in modulation.items()
                if isinstance(value, (int, float))
            }
        else:
            state = {}
        self._modulation_state = state
        if hasattr(self.synapses, "update_modulation"):
            self.synapses.update_modulation(state if state else None)
        if hasattr(self.neurons, "update_modulation") and state:
            try:  # pragma: no cover - optional integration hook
                self.neurons.update_modulation(state)
            except Exception:
                pass

    @property
    def modulation_state(self) -> Dict[str, float]:
        return dict(self._modulation_state)

    def _run_internal(
        self,
        input_events,
        encoder=None,
        *,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        max_duration: int | None = None,
        convergence_threshold: float | None = None,
        convergence_window: int | None = None,
        convergence_patience: int | None = None,
    ):
        queue = EventQueue()
        self.energy_usage = 0
        self.idle_skipped_cycles = 0
        encoder_kwargs = dict(encoder_kwargs or {})
        if encoder is not None:
            for t, analog in enumerate(input_events):
                if not self.idle_skip or any(analog):
                    for time, inputs in encoder(
                        analog, t_start=t, **encoder_kwargs
                    ):
                        queue.push(time, inputs)
                else:
                    self.idle_skipped_cycles += 1
        elif input_events and (
            not isinstance(input_events[0], tuple)
            or len(input_events[0]) != 2
            or not isinstance(input_events[0][0], (int, float))
        ):
            for t, inputs in enumerate(input_events):
                if not self.idle_skip or any(inputs):
                    queue.push(t, inputs)
                else:
                    self.idle_skipped_cycles += 1
        else:
            for t, inputs in input_events:
                if not self.idle_skip or any(inputs):
                    queue.push(t, inputs)
                else:
                    self.idle_skipped_cycles += 1

        outputs: list[tuple[float, list[int]]] = []

        processed = 0
        configured_max = max_duration if max_duration is not None else self.max_duration
        if configured_max is not None:
            max_events = max(1, int(configured_max))
        else:
            max_events = max(32, self.neurons.size * 32)

        threshold = (
            float(convergence_threshold)
            if convergence_threshold is not None
            else (
                float(self.convergence_threshold)
                if self.convergence_threshold is not None
                else None
            )
        )
        window = (
            int(convergence_window)
            if convergence_window is not None
            else self.convergence_window
        )
        if window is not None and window <= 0:
            window = None
        patience = (
            int(convergence_patience)
            if convergence_patience is not None
            else self.convergence_patience
        )
        if patience is None or patience <= 0:
            patience = 1

        recent_spike_totals = (
            deque(maxlen=window) if threshold is not None and window else None
        )
        low_activity_streak = 0

        while queue:
            if processed >= max_events:
                break
            time, inputs = queue.pop()
            self.energy_usage += 1
            spikes = self.neurons.step(inputs)
            for idx, spike in enumerate(spikes):
                if spike:
                    self.spike_times[idx] = time

            if any(spikes):
                self.synapses.adapt(self.spike_times, self.spike_times)

            outputs.append((time, spikes))
            processed += 1

            should_stop = False
            if recent_spike_totals is not None:
                recent_spike_totals.append(sum(spikes))
                if (
                    recent_spike_totals.maxlen
                    and len(recent_spike_totals) == recent_spike_totals.maxlen
                ):
                    avg_spikes = sum(recent_spike_totals) / float(
                        recent_spike_totals.maxlen
                    )
                    if avg_spikes <= (threshold or 0.0):
                        low_activity_streak += 1
                    else:
                        low_activity_streak = 0
                    if low_activity_streak >= patience:
                        should_stop = True

            if should_stop:
                break

            currents = self.synapses.propagate(spikes)
            if any(currents):
                queue.push(time + 1, currents)

        outputs.sort(key=lambda x: x[0])
        return outputs

    def run(
        self,
        input_events,
        encoder=None,
        *,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        max_duration: int | None = None,
        convergence_threshold: float | None = None,
        convergence_window: int | None = None,
        convergence_patience: int | None = None,
        neuromodulation: Mapping[str, float] | None = None,
    ):
        """Run the network using an event-driven simulation."""
        if neuromodulation is not None:
            self.apply_modulation(neuromodulation)
        return self._run_internal(
            input_events,
            encoder,
            encoder_kwargs=encoder_kwargs,
            max_duration=max_duration,
            convergence_threshold=convergence_threshold,
            convergence_window=convergence_window,
            convergence_patience=convergence_patience,
        )

    async def run_async(
        self,
        input_events,
        encoder=None,
        *,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        max_duration: int | None = None,
        convergence_threshold: float | None = None,
        convergence_window: int | None = None,
        convergence_patience: int | None = None,
    ):
        """Asynchronously run the network using ``asyncio``."""
        return await asyncio.to_thread(
            self._run_internal,
            input_events,
            encoder,
            encoder_kwargs=encoder_kwargs,
            max_duration=max_duration,
            convergence_threshold=convergence_threshold,
            convergence_window=convergence_window,
            convergence_patience=convergence_patience,
        )

@dataclass
class NeuromorphicRunResult:
    """Container for spike outputs and derived telemetry."""

    spike_events: List[tuple[float, List[int]]]
    energy_used: float
    idle_skipped: int
    spike_counts: List[int] = field(default_factory=list)
    average_rate: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def spikes(self) -> List[List[int]]:
        return [spike for _, spike in self.spike_events]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spike_events": self.spike_events,
            "energy_used": self.energy_used,
            "idle_skipped": self.idle_skipped,
            "spike_counts": list(self.spike_counts),
            "average_rate": list(self.average_rate),
            "metadata": dict(self.metadata),
        }


class NeuromorphicBackend:
    """Reusable wrapper around :class:`SpikingNeuralNetwork`."""

    _ENCODERS: Dict[str, Callable[..., List[tuple[float, List[int]]]]] = {}

    def __init__(
        self,
        *,
        config: SpikingNetworkConfig | None = None,
        network: SpikingNeuralNetwork | None = None,
        auto_reset: bool = True,
    ) -> None:
        if network is None:
            if config is None:
                raise ValueError("Either config or network must be provided")
            network = config.create()
        self.config = config
        self.network = network
        self.auto_reset = auto_reset
        self._last_modulation: Dict[str, float] = {}
        if not NeuromorphicBackend._ENCODERS:
            NeuromorphicBackend._ENCODERS = {
                "latency": self._latency_encoder,
                "rate": self._rate_encoder,
            }

    @staticmethod
    def _latency_encoder(signal: Sequence[float], *, t_start: float = 0.0, t_scale: float = 1.0) -> List[tuple[float, List[int]]]:
        return latency_encode(list(signal), t_start=t_start, t_scale=t_scale)

    @staticmethod
    def _rate_encoder(signal: Sequence[float], *, steps: int = 5, t_start: float = 0.0) -> List[tuple[float, List[int]]]:
        trains = rate_encode(signal, steps=steps)
        return [(t_start + idx, spikes) for idx, spikes in enumerate(trains)]

    def clone(self) -> "NeuromorphicBackend":
        if self.config is None:
            raise ValueError("Cannot clone backend without original config")
        return NeuromorphicBackend(config=self.config, auto_reset=self.auto_reset)

    def reset_state(self) -> None:
        self.network.reset_state()
        if self._last_modulation:
            self.network.apply_modulation(self._last_modulation)

    def run_events(
        self,
        events,
        *,
        encoder: Callable[..., List[tuple[float, List[int]]]] | None = None,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        decoder: str | None = "counts",
        decoder_kwargs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        neuromodulation: Optional[Mapping[str, float]] = None,
        reset: Optional[bool] = None,
        max_duration: int | None = None,
        convergence_threshold: float | None = None,
        convergence_window: int | None = None,
        convergence_patience: int | None = None,
    ) -> NeuromorphicRunResult:
        if reset or (reset is None and self.auto_reset):
            self.reset_state()
        encoder_kwargs = dict(encoder_kwargs or {})
        modulation = None
        if neuromodulation is not None:
            modulation = {
                key: float(value)
                for key, value in neuromodulation.items()
                if isinstance(value, (int, float))
            }
            self._last_modulation = dict(modulation)
        elif self._last_modulation:
            modulation = dict(self._last_modulation)
        outputs = self.network.run(
            events,
            encoder=encoder,
            encoder_kwargs=encoder_kwargs,
            max_duration=max_duration,
            convergence_threshold=convergence_threshold,
            convergence_window=convergence_window,
            convergence_patience=convergence_patience,
            neuromodulation=modulation,
        )
        counts: List[int] = []
        rates: List[float] = []
        if decoder:
            key = decoder.lower()
            decoder_kwargs = decoder_kwargs or {}
            if key in {"counts", "all"}:
                counts = decode_spike_counts(outputs)
            if key in {"rate", "all"}:
                window = decoder_kwargs.get("window")
                if window is None:
                    window = len(outputs) or 1
                rates = decode_average_rate(outputs, window=float(window))
        result_metadata = dict(metadata or {})
        if modulation:
            result_metadata.setdefault("neuromodulation", dict(modulation))
        synapse_state = getattr(self.network.synapses, "modulation_cache", None)
        if synapse_state:
            result_metadata.setdefault("synapse_modulation", dict(synapse_state))
        return NeuromorphicRunResult(
            spike_events=outputs,
            energy_used=self.network.energy_usage,
            idle_skipped=self.network.idle_skipped_cycles,
            spike_counts=counts,
            average_rate=rates,
            metadata=result_metadata,
        )

    def run_sequence(
        self,
        signal,
        *,
        encoding: str | None = None,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        decoder: str | None = "counts",
        decoder_kwargs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        neuromodulation: Optional[Mapping[str, float]] = None,
        reset: Optional[bool] = None,
        max_duration: int | None = None,
        convergence_threshold: float | None = None,
        convergence_window: int | None = None,
        convergence_patience: int | None = None,
    ) -> NeuromorphicRunResult:
        encoder = None
        encoder_kwargs = dict(encoder_kwargs or {})
        prepared = signal
        if encoding:
            key = encoding.lower()
            if key not in self._ENCODERS:
                raise ValueError(f"Unsupported encoding '{encoding}'")
            encoder = self._ENCODERS[key]
        return self.run_events(
            prepared,
            encoder=encoder,
            encoder_kwargs=encoder_kwargs,
            decoder=decoder,
            decoder_kwargs=decoder_kwargs,
            metadata=metadata,
            neuromodulation=neuromodulation,
            reset=reset,
            max_duration=max_duration,
            convergence_threshold=convergence_threshold,
            convergence_window=convergence_window,
            convergence_patience=convergence_patience,
        )

    def run_batch(
        self,
        sequences,
        *,
        encoding: str | None = None,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        decoder: str | None = "counts",
        decoder_kwargs: Optional[Dict[str, Any]] = None,
        reset_each: bool | None = None,
        neuromodulation: Optional[Mapping[str, float]] = None,
        max_duration: int | None = None,
        convergence_threshold: float | None = None,
        convergence_window: int | None = None,
        convergence_patience: int | None = None,
    ) -> List[NeuromorphicRunResult]:
        results: List[NeuromorphicRunResult] = []
        for sequence in sequences:
            results.append(
                self.run_sequence(
                    sequence,
                    encoding=encoding,
                    encoder_kwargs=encoder_kwargs,
                    decoder=decoder,
                    decoder_kwargs=decoder_kwargs,
                    metadata=None,
                    neuromodulation=neuromodulation,
                    reset=reset_each if reset_each is not None else True,
                    max_duration=max_duration,
                    convergence_threshold=convergence_threshold,
                    convergence_window=convergence_window,
                    convergence_patience=convergence_patience,
                )
            )
        return results


