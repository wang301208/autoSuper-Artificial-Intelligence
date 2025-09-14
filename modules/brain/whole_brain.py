"""High level integration of simplified brain modules.

This module wires together the sensory, cognitive, emotional, conscious and
motor components defined in the surrounding package. The implementation is
deliberately light‑weight – the goal is simply to demonstrate how information
might flow through the different subsystems in a single processing cycle.

The :class:`WholeBrainSimulation` class exposes a :meth:`process_cycle` method
which accepts a dictionary of input data and returns a dictionary describing
the executed action along with basic energy accounting information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from schemas.emotion import EmotionType

from .sensory_cortex import VisualCortex, AuditoryCortex, SomatosensoryCortex
from .motor_cortex import MotorCortex
from .limbic import LimbicSystem
from .consciousness import ConsciousnessModel
from .neuromorphic.spiking_network import SpikingNeuralNetwork


class CognitiveModule:
    """Extremely small placeholder cognitive system.

    The module decides on a high level intention based on the emotional state.
    In a more elaborate implementation this would incorporate the rich
    perception data as well, but for demonstration purposes the emotion is
    sufficient.
    """

    def decide(self, perception: Dict[str, Any], emotion: EmotionType) -> str:
        if emotion == EmotionType.HAPPY:
            return "approach"
        if emotion == EmotionType.SAD:
            return "withdraw"
        return "observe"


@dataclass
class WholeBrainSimulation:
    """Container object coordinating all brain subsystems."""

    visual: VisualCortex = field(default_factory=VisualCortex)
    auditory: AuditoryCortex = field(default_factory=AuditoryCortex)
    somatosensory: SomatosensoryCortex = field(default_factory=SomatosensoryCortex)
    cognition: CognitiveModule = field(default_factory=CognitiveModule)
    emotion: LimbicSystem = field(default_factory=LimbicSystem)
    consciousness: ConsciousnessModel = field(default_factory=ConsciousnessModel)
    motor: MotorCortex = field(default_factory=MotorCortex)
    neuromorphic: bool = True
    last_perception: Dict[str, Any] = field(init=False, default_factory=dict)

    def process_cycle(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single perception‑cognition‑action cycle.

        Parameters
        ----------
        input_data:
            Dictionary that may contain ``image``, ``sound``, ``touch`` and
            ``text`` keys as well as an ``is_salient`` flag indicating whether
            the resulting intention should reach consciousness.

        Returns
        -------
        dict
            Dictionary containing the executed ``action``, total ``energy_used``
            and number of ``idle_skipped`` cycles.
        """

        # --- Sensory processing -------------------------------------------------
        perception = {}
        energy_used = 0
        idle_skipped = 0
        if self.neuromorphic:
            def _encode_and_run(signal: Any, key: str) -> None:
                nonlocal energy_used, idle_skipped
                n = len(signal)
                weights = [[0.0] * n for _ in range(n)]
                network = SpikingNeuralNetwork(
                    n_neurons=n, weights=weights, idle_skip=True, threshold=0.5
                )
                outputs = network.run([signal])
                spike_counts = [0] * n
                for _, spikes in outputs:
                    for i, spike in enumerate(spikes):
                        spike_counts[i] += spike
                perception[key] = {"spike_counts": spike_counts}
                energy_used += network.energy_usage
                idle_skipped += network.idle_skipped_cycles

            if "image" in input_data:
                _encode_and_run(input_data["image"], "vision")
            if "sound" in input_data:
                _encode_and_run(input_data["sound"], "audio")
            if "touch" in input_data:
                _encode_and_run(input_data["touch"], "touch")
        else:
            if "image" in input_data:
                perception["vision"] = self.visual.process(input_data["image"])
            if "sound" in input_data:
                perception["audio"] = self.auditory.process(input_data["sound"])
            if "touch" in input_data:
                perception["touch"] = self.somatosensory.process(input_data["touch"])

        # --- Emotional appraisal ------------------------------------------------
        text_stimulus = input_data.get("text", "")
        emotional_state = self.emotion.react(text_stimulus)

        # --- Cognitive decision making -----------------------------------------
        intention = self.cognition.decide(perception, emotional_state.emotion)

        # --- Conscious access ---------------------------------------------------
        self.consciousness.conscious_access(
            {"is_salient": input_data.get("is_salient", False), "intention": intention}
        )

        # --- Motor execution ----------------------------------------------------
        plan = self.motor.plan_movement(intention)
        action = self.motor.execute_action(plan)
        self.last_perception = perception
        return {"action": action, "energy_used": energy_used, "idle_skipped": idle_skipped}


__all__ = ["WholeBrainSimulation"]

