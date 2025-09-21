import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.neuromorphic import SpikingNeuralNetwork, AdExNeuronModel
from modules.brain.neuromorphic.spiking_network import SpikingNetworkConfig


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


def test_adex_neuron_spiking():
    network = SpikingNeuralNetwork(
        n_neurons=1,
        neuron_model_cls=AdExNeuronModel,
        neuron_model_kwargs={"v_reset": -65.0, "v_threshold": -55.0, "v_peak": 0.0},
        plasticity_mode=None,
        weights=[[0.0]],
    )
    network.synapses.adapt = lambda *args, **kwargs: None
    spikes = network.run([[5.0], [5.0], [5.0]])
    assert any(spike for _, spike in spikes)


def test_spiking_network_config_builder():
    config = SpikingNetworkConfig(
        n_neurons=2,
        neuron="lif",
        neuron_params={"threshold": 0.8},
        weights=[[0.0, 0.5], [0.0, 0.0]],
        plasticity="none",
    )
    network = config.create()
    network.synapses.adapt = lambda *args, **kwargs: None
    outputs = network.run([[1.0, 0.0], [0.0, 0.0]])
    assert outputs
