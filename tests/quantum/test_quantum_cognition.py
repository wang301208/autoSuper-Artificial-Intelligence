import os
import sys

import numpy as np
import pytest

# Ensure the repository root is on the import path when the test module is
# executed in isolation.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.quantum.quantum_cognition import (
    EntanglementNetwork,
    QuantumCognition,
    SuperpositionState,
)


def test_superposition_probabilities():
    state = SuperpositionState({"0": 1 / np.sqrt(2), "1": 1 / np.sqrt(2)})
    qc = QuantumCognition()
    probs = qc.evaluate_probabilities(state)
    assert probs["0"] == pytest.approx(0.5, rel=1e-6)
    assert probs["1"] == pytest.approx(0.5, rel=1e-6)


def test_entangled_density_matrix_probabilities():
    network = EntanglementNetwork()
    bell_state = network.create_bell_pair()
    density = bell_state.density_matrix()
    qc = QuantumCognition()
    probs = qc.evaluate_probabilities(density)
    assert probs["00"] == pytest.approx(0.5, rel=1e-6)
    assert probs["11"] == pytest.approx(0.5, rel=1e-6)
    assert probs["01"] == pytest.approx(0.0, abs=1e-6)
    assert probs["10"] == pytest.approx(0.0, abs=1e-6)
