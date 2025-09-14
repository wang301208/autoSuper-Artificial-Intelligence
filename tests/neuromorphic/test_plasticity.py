import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.neuromorphic import SpikingNeuralNetwork


def test_stdp_potentiation():
    weights = [[0.0, 0.0], [0.0, 0.0]]
    network = SpikingNeuralNetwork(n_neurons=2, threshold=1.0, reset=0.0, decay=1.0, weights=weights)
    inputs = [[1.1, 0.0], [0.0, 1.1]]  # pre neuron fires before post neuron
    network.run(inputs)
    assert network.synapses.weights[0][1] > 0.0


def test_stdp_depression():
    weights = [[0.0, 0.0], [0.0, 0.0]]
    network = SpikingNeuralNetwork(n_neurons=2, threshold=1.0, reset=0.0, decay=1.0, weights=weights)
    inputs = [[0.0, 1.1], [1.1, 0.0]]  # post neuron fires before pre neuron
    network.run(inputs)
    assert network.synapses.weights[0][1] < 0.0
