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


def test_refractory_behavior():
    neurons = SpikingNeuralNetwork.LeakyIntegrateFireNeurons(
        n_neurons=1, decay=1.0, threshold=1.0, reset=0.0, refractory_period=2
    )

    assert neurons.step([1.1]) == [1]
    assert neurons.step([1.1]) == [0]
    assert neurons.step([1.1]) == [0]
    assert neurons.step([1.1]) == [1]


def test_dynamic_threshold_adaptation():
    neurons = SpikingNeuralNetwork.LeakyIntegrateFireNeurons(
        n_neurons=1,
        decay=0.9,
        threshold=1.0,
        reset=0.0,
        dynamic_threshold=0.5,
    )

    # Initial spike raises threshold
    assert neurons.step([1.1]) == [1]
    # Elevated threshold suppresses subsequent spike
    assert neurons.step([1.1]) == [0]
    # Allow adaptive threshold to decay
    for _ in range(20):
        neurons.step([0.0])
    # Same input can trigger spike again after adaptation decays
    assert neurons.step([1.1]) == [1]
