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
        self.learning_rate = learning_rate
        self._plasticity_cls = plasticity.__class__ if plasticity is not None else None
        self.plasticity = plasticity if plasticity is not None else Neuroplasticity()

    def reset_state(self) -> None:
        self.weights = self._initial_weights.copy()
        if self._plasticity_cls is not None:
            self.plasticity = self._plasticity_cls()

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
        self.idle_skip = idle_skip
        self.energy_usage = 0
        self.idle_skipped_cycles = 0

    def reset_state(self) -> None:
        if hasattr(self.neurons, "reset_state"):
            self.neurons.reset_state()
        if hasattr(self.synapses, "reset_state"):
            self.synapses.reset_state()
        self.spike_times = [None] * len(self.spike_times)
        self.energy_usage = 0
        self.idle_skipped_cycles = 0

    def _run_internal(self, input_events, encoder=None, **encoder_kwargs):
        queue = EventQueue()
        self.energy_usage = 0
        self.idle_skipped_cycles = 0
        if encoder is not None:
            for t, analog in enumerate(input_events):
                if not self.idle_skip or any(analog):
                    for time, inputs in encoder(analog, t_start=t, **encoder_kwargs):
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
        max_events = max(32, self.neurons.size * 32)
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

            currents = self.synapses.propagate(spikes)
            if any(currents):
                queue.push(time + 1, currents)

            processed += 1

        outputs.sort(key=lambda x: x[0])
        return outputs

    def run(self, input_events, encoder=None, **encoder_kwargs):
        """Run the network using an event-driven simulation."""
        return self._run_internal(input_events, encoder, **encoder_kwargs)

    async def run_async(self, input_events, encoder=None, **encoder_kwargs):
        """Asynchronously run the network using ``asyncio``."""
        return await asyncio.to_thread(
            self._run_internal, input_events, encoder, **encoder_kwargs
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

    def run_events(
        self,
        events,
        *,
        encoder: Callable[..., List[tuple[float, List[int]]]] | None = None,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        decoder: str | None = "counts",
        decoder_kwargs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        reset: Optional[bool] = None,
    ) -> NeuromorphicRunResult:
        if reset or (reset is None and self.auto_reset):
            self.reset_state()
        encoder_kwargs = encoder_kwargs or {}
        outputs = self.network.run(events, encoder=encoder, **encoder_kwargs)
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
        return NeuromorphicRunResult(
            spike_events=outputs,
            energy_used=self.network.energy_usage,
            idle_skipped=self.network.idle_skipped_cycles,
            spike_counts=counts,
            average_rate=rates,
            metadata=metadata or {},
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
        reset: Optional[bool] = None,
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
            reset=reset,
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
                    reset=reset_each if reset_each is not None else True,
                )
            )
        return results


