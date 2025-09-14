"""Quantum cognition components.

This module provides lightweight representations of quantum states and a
simple network for generating entangled states.  It purposely avoids any
heavy dependencies or advanced simulation features; the goal is merely to
support unit tests that exercise basic superposition and measurement
behaviour.

Classes
-------
SuperpositionState
    Stores complex amplitudes for basis states and exposes utilities to
    obtain probabilities or a density matrix representation.
EntanglementNetwork
    Produces a small set of predefined entangled states used by the tests.
QuantumCognition
    High level interface exposing :meth:`evaluate_probabilities` which can
    operate on either :class:`SuperpositionState` instances or density
    matrices.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Union

import numpy as np


@dataclass
class SuperpositionState:
    """Representation of a quantum superposition.

    Parameters
    ----------
    amplitudes:
        Mapping of basis state labels to complex amplitudes.  The
        amplitudes are normalised on construction so their squared
        magnitudes sum to one.
    """

    amplitudes: Dict[str, complex]

    def __post_init__(self) -> None:
        norm = math.sqrt(sum(abs(a) ** 2 for a in self.amplitudes.values()))
        if not np.isclose(norm, 1):
            self.amplitudes = {k: v / norm for k, v in self.amplitudes.items()}

    def probabilities(self) -> Dict[str, float]:
        """Return measurement probabilities for each basis state."""
        return {k: float(abs(a) ** 2) for k, a in self.amplitudes.items()}

    def density_matrix(self) -> np.ndarray:
        """Return the density matrix corresponding to the state."""
        labels = sorted(self.amplitudes.keys())
        vector = np.array([self.amplitudes[l] for l in labels], dtype=complex)
        return np.outer(vector, np.conjugate(vector))


@dataclass
class EntanglementNetwork:
    """Utility for creating simple entangled states."""

    def create_bell_pair(self) -> SuperpositionState:
        """Return a Bell pair :math:`(|00\rangle + |11\rangle)/\sqrt{2}`."""
        amp = 1 / math.sqrt(2)
        # Include the full two-qubit computational basis with zero amplitudes
        return SuperpositionState({"00": amp, "01": 0, "10": 0, "11": amp})


class QuantumCognition:
    """High level interface for evaluating quantum cognitive states."""

    def __init__(self, network: EntanglementNetwork | None = None) -> None:
        self.network = network or EntanglementNetwork()

    def evaluate_probabilities(
        self, input_state: Union[SuperpositionState, np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate measurement probabilities for *input_state*.

        ``input_state`` may either be a :class:`SuperpositionState` or a
        density matrix expressed as a NumPy array.  In the latter case the
        diagonal entries of the matrix are interpreted as probabilities in
        the computational basis.
        """

        if isinstance(input_state, SuperpositionState):
            return input_state.probabilities()

        matrix = np.asarray(input_state, dtype=complex)
        diag = np.real_if_close(np.diag(matrix))
        size = diag.shape[0]
        num_qubits = int(math.log2(size)) if size else 0
        labels = [format(i, f"0{num_qubits}b") for i in range(size)]
        return {label: float(diag[i]) for i, label in enumerate(labels)}


__all__ = ["SuperpositionState", "EntanglementNetwork", "QuantumCognition"]
