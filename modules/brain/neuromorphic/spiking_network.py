class SpikingNeuralNetwork:
    """A minimal spiking neural network with leaky integrate-and-fire neurons.

    The network consists of a population of LIF neurons connected by dynamic
    synapses. The synapses expose a simple plasticity interface compatible with
    spike timing dependent plasticity (STDP) style rules.
    """

    class LeakyIntegrateFireNeurons:
        """Set of leaky integrate-and-fire neurons."""

        def __init__(self, n_neurons, decay=0.9, threshold=1.0, reset=0.0):
            self.potentials = [0.0] * n_neurons
            self.decay = decay
            self.threshold = threshold
            self.reset = reset

        def step(self, inputs):
            """Update membrane potentials given input currents.

            Returns a list of spikes (1 or 0) indicating whether each neuron
            fired on this step. Potentials are reset after spiking.
            """
            spikes = []
            for i, current in enumerate(inputs):
                v = self.potentials[i] * self.decay + current
                if v >= self.threshold:
                    spikes.append(1)
                    self.potentials[i] = self.reset
                else:
                    spikes.append(0)
                    self.potentials[i] = v
            return spikes

    class DynamicSynapses:
        """Synaptic connections with a simple plasticity interface."""

        def __init__(self, weights):
            self.weights = weights

        def propagate(self, pre_spikes):
            """Propagate spikes through the synapses to produce currents."""
            postsynaptic = [0.0 for _ in range(len(self.weights[0]))]
            for j, pre in enumerate(pre_spikes):
                for i, weight in enumerate(self.weights[j]):
                    postsynaptic[i] += pre * weight
            return postsynaptic

        def adapt(self, pre_spikes, post_spikes, learning_rate=0.1):
            """Update synaptic weights using a Hebbian-like rule.

            This simple rule is compatible with STDP formulations and serves as
            a placeholder for more sophisticated plasticity mechanisms.
            """
            for j, pre in enumerate(pre_spikes):
                for i, post in enumerate(post_spikes):
                    self.weights[j][i] += learning_rate * pre * post

    def __init__(self, n_neurons, decay=0.9, threshold=1.0, reset=0.0, weights=None):
        if weights is None:
            weights = [[1.0] * n_neurons for _ in range(n_neurons)]
        self.neurons = self.LeakyIntegrateFireNeurons(n_neurons, decay, threshold, reset)
        self.synapses = self.DynamicSynapses(weights)

    def run(self, input_sequence):
        """Run the network for a sequence of inputs.

        Each element in ``input_sequence`` should be a list of input currents for
        all neurons at one timestep. Returns a list of spike vectors emitted at
        each timestep.
        """
        outputs = []
        for inputs in input_sequence:
            currents = self.synapses.propagate(inputs)
            spikes = self.neurons.step(currents)
            # The adapt call exposes an interface for plasticity rules like STDP.
            self.synapses.adapt(inputs, spikes)
            outputs.append(spikes)
        return outputs
