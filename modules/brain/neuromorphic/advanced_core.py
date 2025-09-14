"""Advanced neuromorphic building blocks for richer network simulation.

This module defines a small collection of neuron and synapse models along with
utilities for generating network topologies and targeting specific hardware
backends.  The implementations are intentionally lightweight – the goal is to
provide simple, easily-instantiated objects suitable for unit testing and basic
simulation experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import random

# ---------------------------------------------------------------------------
# Neuron models
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveExponentialIF:
    """Adaptive exponential integrate-and-fire neuron.

    This extremely small model captures only the essence of the AdEx neuron: a
    membrane potential ``v`` that integrates input current and an adaptation
    variable ``a`` that increases each time the neuron spikes.  When ``v``
    exceeds the threshold (``v_th`` + ``a``) a spike is emitted, ``v`` is reset
    and the adaptation term is incremented.
    """

    v: float = 0.0
    tau_m: float = 20.0
    v_th: float = 1.0
    a: float = 0.0
    tau_w: float = 100.0

    def step(self, current: float) -> int:
        self.v += (-self.v + current) / self.tau_m
        self.a += -self.a / self.tau_w
        if self.v >= self.v_th + self.a:
            self.v = 0.0
            self.a += 0.5
            return 1
        return 0


@dataclass
class FastSpikingInterneuron:
    """Minimal fast-spiking interneuron model."""

    v: float = 0.0
    v_th: float = 1.0

    def step(self, current: float) -> int:
        self.v += current
        if self.v >= self.v_th:
            self.v = 0.0
            return 1
        return 0


@dataclass
class DopamineNeuron:
    """Integrate-and-fire neuron that tracks dopamine release."""

    v: float = 0.0
    dopamine: float = 0.0

    def step(self, current: float) -> int:
        self.v += current
        if self.v >= 1.0:
            self.v = 0.0
            self.dopamine += 1.0
            return 1
        return 0


@dataclass
class ChatteringNeuron:
    """Simple bursting neuron that emits two spikes on firing."""

    v: float = 0.0

    def step(self, current: float) -> int:
        self.v += current
        if self.v >= 1.0:
            self.v = 0.0
            # Emit a burst represented by two spikes
            return 2
        return 0


# ---------------------------------------------------------------------------
# Synapse models
# ---------------------------------------------------------------------------


@dataclass
class AMPAReceptor:
    """Excitatory AMPA receptor."""

    weight: float = 1.0

    def transmit(self, spike: int) -> float:
        return self.weight * spike


@dataclass
class GABAReceptor:
    """Inhibitory GABA receptor."""

    weight: float = -1.0

    def transmit(self, spike: int) -> float:
        return self.weight * spike


@dataclass
class DopamineReceptor:
    """Neuromodulatory dopamine receptor."""

    weight: float = 1.0
    modulation: float = 0.0

    def transmit(self, spike: int) -> float:
        return (self.weight + self.modulation) * spike


@dataclass
class CholinergicReceptor:
    """Excitatory cholinergic receptor."""

    weight: float = 1.0

    def transmit(self, spike: int) -> float:
        return self.weight * spike


# ---------------------------------------------------------------------------
# Network topology generation
# ---------------------------------------------------------------------------


class NetworkTopologyGenerator:
    """Generate basic network topologies.

    The generator supports three simple topologies: small-world, scale-free and
    modular.  Each method returns an adjacency matrix represented as a nested
    list ``matrix[i][j]`` where non-zero values indicate a connection from node
    ``i`` to ``j``.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.random = random.Random(seed)

    # Public API -----------------------------------------------------------
    def generate(self, n: int, topology: str, **params) -> List[List[int]]:
        if topology == "small_world":
            return self._small_world(n, params.get("k", 4), params.get("p", 0.1))
        if topology == "scale_free":
            return self._scale_free(n, params.get("m", 2))
        if topology == "modular":
            return self._modular(
                n,
                params.get("modules", 2),
                params.get("p_in", 0.8),
                params.get("p_out", 0.05),
            )
        raise ValueError(f"Unknown topology: {topology}")

    # Internal helpers ----------------------------------------------------
    def _small_world(self, n: int, k: int, p: float) -> List[List[int]]:
        # Start with ring lattice
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(1, k // 2 + 1):
                matrix[i][(i + j) % n] = 1
                matrix[i][(i - j) % n] = 1
        # Rewire edges
        for i in range(n):
            for j in range(1, k // 2 + 1):
                if self.random.random() < p:
                    target = self.random.randrange(n)
                    matrix[i][(i + j) % n] = 0
                    matrix[i][target] = 1
        return matrix

    def _scale_free(self, n: int, m: int) -> List[List[int]]:
        m = max(1, m)
        # Start with a fully connected core of m+1 nodes
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        degrees = [0] * n
        for i in range(m + 1):
            for j in range(m + 1):
                if i != j:
                    matrix[i][j] = 1
                    degrees[i] += 1
        for new_node in range(m + 1, n):
            # Preferential attachment
            targets = self._weighted_choice(range(new_node), degrees[:new_node], m)
            for t in targets:
                matrix[new_node][t] = 1
                matrix[t][new_node] = 1
                degrees[new_node] += 1
                degrees[t] += 1
        return matrix

    def _modular(self, n: int, modules: int, p_in: float, p_out: float) -> List[List[int]]:
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        sizes = [n // modules] * modules
        for i in range(n % modules):
            sizes[i] += 1
        # Assign nodes to modules
        assignments: List[int] = []
        for idx, size in enumerate(sizes):
            assignments.extend([idx] * size)
        for i in range(n):
            for j in range(i + 1, n):
                if assignments[i] == assignments[j]:
                    prob = p_in
                else:
                    prob = p_out
                if self.random.random() < prob:
                    matrix[i][j] = matrix[j][i] = 1
        return matrix

    def _weighted_choice(self, choices, weights, k):
        total = sum(weights)
        if total == 0:
            return self.random.sample(list(choices), k)
        selected = []
        for _ in range(k):
            r = self.random.uniform(0, total)
            upto = 0
            for choice, weight in zip(choices, weights):
                if upto + weight >= r:
                    selected.append(choice)
                    break
                upto += weight
        return selected


# ---------------------------------------------------------------------------
# Hardware backend
# ---------------------------------------------------------------------------


@dataclass
class NeuromorphicHardwareBackend:
    """Abstraction for targeting neuromorphic hardware."""

    target_chip: str = "loihi"
    optimization_level: int = 0

    def __post_init__(self) -> None:
        if self.target_chip.lower() != "loihi":
            raise ValueError(f"Unsupported chip: {self.target_chip}")

    def compile(self, network) -> Dict[str, int]:  # pragma: no cover - trivial
        """Pretend to compile ``network`` for the given hardware."""
        return {"compiled": 1, "optimization_level": self.optimization_level}


# ---------------------------------------------------------------------------
# Core wrapper
# ---------------------------------------------------------------------------


class AdvancedNeuromorphicCore:
    """Facade bundling neurons, synapses, topology generation and backend."""

    neuron_types = {
        "adaptive_exponential": AdaptiveExponentialIF,
        "fast_spiking": FastSpikingInterneuron,
        "dopamine": DopamineNeuron,
        "chattering": ChatteringNeuron,
    }

    synapse_types = {
        "ampa": AMPAReceptor,
        "gaba": GABAReceptor,
        "dopamine": DopamineReceptor,
        "cholinergic": CholinergicReceptor,
    }

    def __init__(self, backend: Optional[NeuromorphicHardwareBackend] = None) -> None:
        self.backend = backend or NeuromorphicHardwareBackend()
        self.topology_generator = NetworkTopologyGenerator()

    def create_neuron(self, neuron_type: str, **params):
        cls = self.neuron_types[neuron_type]
        return cls(**params)

    def create_synapse(self, synapse_type: str, **params):
        cls = self.synapse_types[synapse_type]
        return cls(**params)

    def generate_topology(self, n: int, topology: str, **params) -> List[List[int]]:
        return self.topology_generator.generate(n, topology, **params)

    def select_backend(self, target_chip: str, optimization_level: int = 0) -> NeuromorphicHardwareBackend:
        self.backend = NeuromorphicHardwareBackend(target_chip, optimization_level)
        return self.backend


__all__ = [
    "AdaptiveExponentialIF",
    "FastSpikingInterneuron",
    "DopamineNeuron",
    "ChatteringNeuron",
    "AMPAReceptor",
    "GABAReceptor",
    "DopamineReceptor",
    "CholinergicReceptor",
    "NetworkTopologyGenerator",
    "NeuromorphicHardwareBackend",
    "AdvancedNeuromorphicCore",
]
