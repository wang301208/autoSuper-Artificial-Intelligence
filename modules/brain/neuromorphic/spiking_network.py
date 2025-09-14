import heapq
import random
from typing import List, Tuple

from modules.brain.neuroplasticity import Neuroplasticity


class EventQueue:
    """Priority queue managing spike events by timestamp."""

    def __init__(self) -> None:
        self._queue: List[Tuple[float, List[float]]] = []

    def push(self, time: float, inputs: List[float]) -> None:
        heapq.heappush(self._queue, (time, inputs))

    def pop(self) -> Tuple[float, List[float]]:
        return heapq.heappop(self._queue)

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return bool(self._queue)


class SpikingNeuralNetwork:
    """A minimal spiking neural network with leaky integrate-and-fire neurons.

    The network consists of a population of LIF neurons connected by dynamic
    synapses. The synapses expose a simple plasticity interface compatible with
    spike timing dependent plasticity (STDP) style rules.
    """

    class LeakyIntegrateFireNeurons:
        """Set of leaky integrate-and-fire neurons."""

        def __init__(
            self,
            n_neurons,
            decay=0.9,
            threshold=1.0,
            reset=0.0,
            refractory_period=0,
            dynamic_threshold=0.0,
            noise=None,
        ):
            self.potentials = [0.0] * n_neurons
            self.decay = decay
            self.threshold = threshold
            self.reset = reset
            self.refractory_period = refractory_period
            self.dynamic_threshold = dynamic_threshold
            self.noise = noise

            # Track per-neuron refractory timers and adaptive thresholds
            self.refractory = [0] * n_neurons
            self.adaptation = [0.0] * n_neurons

        def step(self, inputs):
            """Update membrane potentials given input currents.

            Returns a list of spikes (1 or 0) indicating whether each neuron
            fired on this step. Potentials are reset after spiking. Neurons
            respect refractory periods and thresholds adapt based on recent
            spiking activity. Optional noise can be injected into the membrane
            potential update.
            """
            spikes = []
            for i, current in enumerate(inputs):
                # Handle refractory state
                if self.refractory[i] > 0:
                    spikes.append(0)
                    self.refractory[i] -= 1
                    self.potentials[i] = self.reset
                    self.adaptation[i] *= self.decay
                    continue

                v = self.potentials[i] * self.decay + current
                if self.noise is not None:
                    v += random.gauss(0, self.noise)

                threshold = self.threshold + self.adaptation[i]
                if v >= threshold:
                    spikes.append(1)
                    self.potentials[i] = self.reset
                    self.refractory[i] = self.refractory_period
                    self.adaptation[i] += self.dynamic_threshold
                else:
                    spikes.append(0)
                    self.potentials[i] = v
                    self.adaptation[i] *= self.decay
            return spikes

    class DynamicSynapses:
        """Synaptic connections with a simple plasticity interface."""

        def __init__(self, weights):
            self.weights = weights
            self.plasticity = Neuroplasticity()

        def propagate(self, pre_spikes):
            """Propagate spikes through the synapses to produce currents."""
            postsynaptic = [0.0 for _ in range(len(self.weights[0]))]
            for j, pre in enumerate(pre_spikes):
                for i, weight in enumerate(self.weights[j]):
                    postsynaptic[i] += pre * weight
            return postsynaptic

        def adapt(self, pre_spike_times, post_spike_times, learning_rate=0.1):
            """Update synaptic weights based on spike timing.

            Delegates the actual plasticity rule to the ``Neuroplasticity``
            module which implements a simple STDP-like update. We ignore pairs
            for which either neuron has not spiked yet (``None`` timestamps).
            """
            for j, pre_time in enumerate(pre_spike_times):
                if pre_time is None:
                    continue
                for i, post_time in enumerate(post_spike_times):
                    if post_time is None:
                        continue
                    delta = self.plasticity.adapt_connections(pre_time, post_time)
                    self.weights[j][i] += learning_rate * delta

    def __init__(
        self,
        n_neurons,
        decay=0.9,
        threshold=1.0,
        reset=0.0,
        weights=None,
        refractory_period=0,
        dynamic_threshold=0.0,
        noise=None,
    ):
        if weights is None:
            weights = [[1.0] * n_neurons for _ in range(n_neurons)]
        self.neurons = self.LeakyIntegrateFireNeurons(
            n_neurons,
            decay,
            threshold,
            reset,
            refractory_period,
            dynamic_threshold,
            noise,
        )
        self.synapses = self.DynamicSynapses(weights)
        self.spike_times = [None] * n_neurons

    def run(self, input_events):
        """Run the network using an event-driven simulation.

        ``input_events`` may be provided either as a sequence of input vectors
        (in which case events are assumed to occur at successive integer
        timestamps starting at zero) or as an iterable of ``(time, inputs)``
        pairs. Each event is processed in temporal order and any spikes produced
        dispatch a new event at ``time + 1`` carrying the postsynaptic currents.

        Returns a list of ``(time, spikes)`` tuples denoting when neurons fired.
        """
        queue = EventQueue()
        # Allow legacy list-of-inputs style by enumerating timestamps
        if input_events and (
            not isinstance(input_events[0], tuple)
            or len(input_events[0]) != 2
            or not isinstance(input_events[0][0], (int, float))
        ):
            for t, inputs in enumerate(input_events):
                queue.push(t, inputs)
        else:
            for t, inputs in input_events:
                queue.push(t, inputs)

        outputs: List[Tuple[float, List[int]]] = []

        while queue:
            time, inputs = queue.pop()
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

        outputs.sort(key=lambda x: x[0])
        return outputs
