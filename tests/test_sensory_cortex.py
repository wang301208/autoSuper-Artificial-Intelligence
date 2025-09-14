import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.brain import VisualCortex, AuditoryCortex, SomatosensoryCortex
from modules.brain.neuromorphic.spiking_network import SpikingNeuralNetwork


def test_visual_cortex():
    cortex = VisualCortex()
    result = cortex.process("image data")
    assert "edges" in result and "color" in result


def test_auditory_cortex():
    cortex = AuditoryCortex()
    result = cortex.process("audio data")
    assert "frequencies" in result and "interpretation" in result


def test_somatosensory_cortex():
    cortex = SomatosensoryCortex()
    result = cortex.process("stimulus")
    assert result == ["touch"]


def test_visual_cortex_spiking_backend():
    snn = SpikingNeuralNetwork(2, weights=[[0.0, 1.0], [0.0, 0.0]])
    initial = snn.synapses.weights[0][1]
    cortex = VisualCortex(spiking_backend=snn)
    cortex.process([[1, 0], [0, 1]])
    assert snn.synapses.weights[0][1] != initial
