import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.neuromorphic import SpikingNeuralNetwork


def test_spike_generation():
    network = SpikingNeuralNetwork(
        n_neurons=1, decay=0.8, threshold=1.0, reset=0.0, weights=[[0.0]]
    )
    network.synapses.adapt = lambda *args, **kwargs: None
    inputs = [[0.6], [0.6], [0.0], [1.2]]
    spikes = network.run(inputs)
    expected = [(0, [0]), (1, [1]), (2, [0]), (3, [1])]
    assert spikes == expected
