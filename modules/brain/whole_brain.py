"""Production-grade integration of the cognitive architecture.

This module orchestrates the sensory, cognitive, emotional, conscious and
motor components defined in the surrounding package.  The implementation now
supports stateful streaming inputs, neuromorphic hardware backends, cognitive
policy pluggability and telemetry suitable for deployment scenarios.

The :class:`WholeBrainSimulation` class exposes a :meth:`process_cycle` method
which accepts structured input data and returns a detailed
``BrainCycleResult`` containing perception, emotion, and action intent
snapshots for downstream agents.
"""

from __future__ import annotations

import logging
import math
import numpy as np
from dataclasses import dataclass, field
from numbers import Real
from collections import OrderedDict, deque
from typing import Any, Dict, Iterable, List, Optional, Tuple

from schemas.emotion import EmotionType

from .sensory_cortex import VisualCortex, AuditoryCortex, SomatosensoryCortex
from .motor_cortex import MotorCortex
from .cerebellum import Cerebellum
from .oscillations import NeuralOscillations
from .limbic import LimbicSystem
from .state import (
    BrainCycleResult,
    BrainRuntimeConfig,
    CognitiveIntent,
    CuriosityState,
    EmotionSnapshot,
    FeelingSnapshot,
    PerceptionSnapshot,
    PersonalityProfile,
    ThoughtSnapshot,
)
from .consciousness import ConsciousnessModel
from .neuromorphic.spiking_network import SpikingNetworkConfig, NeuromorphicBackend, NeuromorphicRunResult
from .self_learning import SelfLearningBrain
from .neuromorphic.temporal_encoding import latency_encode, rate_encode, decode_spike_counts
from .perception import SensoryPipeline, EncodedSignal
from .motor.precision import PrecisionMotorSystem
from .motor.actions import MotorExecutionResult, MotorPlan


logger = logging.getLogger(__name__)


def default_plan_for_intention(
    intention: str,
    focus: Optional[str],
    context: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Generate a lightweight plan for a given intention."""

    context = context or {}
    plan: List[str] = []
    if intention == "explore":
        plan = [
            "scan_environment",
            f"focus_{focus}" if focus else "sample_new_modalities",
            "log_novelty",
        ]
    elif intention == "approach":
        plan = [
            "identify_positive_stimulus",
            f"move_towards_{focus}" if focus else "establish_focus",
            "engage",
        ]
    elif intention == "withdraw":
        plan = ["assess_risk", "increase_distance", "seek_support"]
    else:
        plan = ["monitor_sensory_streams", "maintain_attention"]

    task = context.get("task")
    if task:
        plan.append(f"respect_task_{task}")
    return [step for step in plan if step]


@dataclass
class CognitiveDecision:
    """Container for policy-driven cognitive decisions."""

    intention: str
    confidence: float
    plan: List[str] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    focus: Optional[str] = None
    summary: str = ""
    thought_trace: List[str] = field(default_factory=list)
    perception_summary: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CognitivePolicy:
    """Interface for pluggable cognitive intention selection policies."""

    def select_intention(
        self,
        perception: PerceptionSnapshot,
        summary: Dict[str, float],
        emotion: EmotionSnapshot,
        personality: PersonalityProfile,
        curiosity: CuriosityState,
        context: Dict[str, Any],
        learning_prediction: Optional[Dict[str, float]] = None,
    ) -> CognitiveDecision:
        raise NotImplementedError


class HeuristicCognitivePolicy(CognitivePolicy):
    """Default heuristic policy mirroring the legacy weighting logic."""

    def _build_tags(
        self,
        intention: str,
        confidence: float,
        curiosity: CuriosityState,
        focus: Optional[str],
    ) -> List[str]:
        tags = [intention]
        if confidence >= 0.65:
            tags.append("high-confidence")
        if curiosity.last_novelty > 0.6:
            tags.append("novelty-driven")
        if focus:
            tags.append(f"focus-{focus}")
        return tags

    def select_intention(
        self,
        perception: PerceptionSnapshot,
        summary: Dict[str, float],
        emotion: EmotionSnapshot,
        personality: PersonalityProfile,
        curiosity: CuriosityState,
        context: Dict[str, Any],
        learning_prediction: Optional[Dict[str, float]] = None,
    ) -> CognitiveDecision:
        focus = max(summary, key=summary.get) if summary else None
        options = {
            "observe": 0.2 + (1 - abs(emotion.dimensions.get("valence", 0.0))) * 0.3,
            "approach": 0.2 + emotion.intent_bias.get("approach", 0.0),
            "withdraw": 0.2 + emotion.intent_bias.get("withdraw", 0.0),
            "explore": 0.2
            + emotion.intent_bias.get("explore", 0.0)
            + curiosity.drive * 0.5,
        }
        if learning_prediction:
            predicted_load = float(learning_prediction.get("cpu", 0.0))
            resource_pressure = float(learning_prediction.get("memory", 0.0))
            options["observe"] += max(0.0, predicted_load - 0.5) * 0.3
            options["withdraw"] += max(0.0, resource_pressure - 0.5) * 0.2
            options["approach"] += max(0.0, 0.5 - predicted_load) * 0.2
        if context.get("threat", 0.0) > 0.4:
            options["withdraw"] += 0.3
        if context.get("safety", 0.0) > 0.5:
            options["approach"] += 0.2
        options["explore"] *= 0.5 + personality.modulation_weight("explore")
        options["approach"] *= 0.5 + personality.modulation_weight("social")
        options["withdraw"] *= 0.5 + personality.modulation_weight("caution")
        options["observe"] *= 0.5 + personality.modulation_weight("persist")
        total = sum(options.values()) or 1.0
        weights = {key: value / total for key, value in options.items()}
        intention = max(weights.items(), key=lambda item: item[1])[0]
        confidence = weights[intention]
        plan = default_plan_for_intention(intention, focus, context)
        tags = self._build_tags(intention, confidence, curiosity, focus)
        thought_trace = [
            f"focus={focus or 'none'}",
            f"intention={intention}",
            f"emotion={emotion.primary.value}:{emotion.intensity:.2f}",
            f"curiosity={curiosity.drive:.2f}",
        ]
        if learning_prediction:
            thought_trace.append(
                f"predicted_cpu={float(learning_prediction.get('cpu', 0.0)):.2f}"
            )
            thought_trace.append(
                f"predicted_mem={float(learning_prediction.get('memory', 0.0)):.2f}"
            )
        summary_text = (
            ", ".join(f"{k}:{v:.2f}" for k, v in summary.items())
            or "no-salient-modalities"
        )
        return CognitiveDecision(
            intention=intention,
            confidence=confidence,
            plan=plan,
            weights=weights,
            tags=tags,
            focus=focus,
            summary=summary_text,
            thought_trace=thought_trace,
            perception_summary=dict(summary),
            metadata={"policy": "heuristic"},
        )


class CognitiveModule:
    """Lightweight cognitive reasoning delegating intention selection to policies."""

    def __init__(
        self,
        memory_window: int = 8,
        policy: Optional[CognitivePolicy] = None,
    ) -> None:
        self.memory_window = memory_window
        self.episodic_memory: deque[dict[str, Any]] = deque(maxlen=memory_window)
        self.policy: CognitivePolicy = policy or HeuristicCognitivePolicy()
        self._confidence_history: deque[float] = deque(maxlen=max(4, memory_window))

    def set_policy(self, policy: CognitivePolicy) -> None:
        """Replace the active policy at runtime."""

        self.policy = policy

    def _summarise_perception(self, perception: PerceptionSnapshot) -> Dict[str, float]:
        summary: Dict[str, float] = {}
        for name, payload in perception.modalities.items():
            spikes = payload.get("spike_counts") or []
            total = float(sum(spikes))
            if total > 0:
                summary[name] = total
        total_sum = sum(summary.values())
        if total_sum <= 0:
            return {name: 0.0 for name in perception.modalities}
        return {name: value / total_sum for name, value in summary.items()}

    def _calibrate_confidence(self, raw_confidence: float) -> float:
        raw = max(0.0, min(1.0, float(raw_confidence)))
        if not self._confidence_history:
            return raw
        mean = sum(self._confidence_history) / len(self._confidence_history)
        if mean <= 0:
            calibrated = raw
        elif raw >= mean:
            calibrated = 0.5 + 0.5 * (raw - mean) / (1 - mean + 1e-6)
        else:
            calibrated = 0.5 * raw / (mean + 1e-6)
        return max(0.0, min(1.0, calibrated))

    def _build_plan(
        self,
        intention: str,
        summary: Dict[str, float],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        focus = max(summary, key=summary.get) if summary else None
        return default_plan_for_intention(intention, focus, context)

    def _remember(self, summary: Dict[str, float], emotion: EmotionSnapshot, intention: str, confidence: float) -> None:
        self.episodic_memory.append(
            {
                "summary": summary,
                "emotion": emotion.primary.value,
                "intensity": emotion.intensity,
                "intention": intention,
                "confidence": confidence,
            }
        )

    def recall(self, limit: int = 5) -> List[dict[str, Any]]:
        if limit <= 0:
            return list(self.episodic_memory)
        return list(self.episodic_memory)[-limit:]



    def decide(
        self,
        perception: PerceptionSnapshot,
        emotion: EmotionSnapshot,
        personality: PersonalityProfile,
        curiosity: CuriosityState,
        learning_prediction: Optional[Dict[str, float]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = context or {}
        summary = self._summarise_perception(perception)
        try:
            policy_decision = self.policy.select_intention(
                perception,
                summary,
                emotion,
                personality,
                curiosity,
                context,
                learning_prediction,
            )
        except Exception as exc:  # pragma: no cover - defensive policy fallback
            logger.warning("Cognitive policy failed; using fallback plan. Error: %s", exc)
            intention = context.get("fallback_intention", "observe")
            policy_decision = CognitiveDecision(
                intention=intention,
                confidence=0.25,
                plan=self._build_plan(intention, summary, context),
                weights={intention: 1.0},
                tags=[intention, "policy-fallback"],
                focus=max(summary, key=summary.get) if summary else None,
                summary=", ".join(f"{k}:{v:.2f}" for k, v in summary.items())
                or "no-salient-modalities",
                thought_trace=["policy=fallback"],
                perception_summary=dict(summary),
                metadata={"policy": "fallback", "error": str(exc)},
            )

        if not policy_decision.plan:
            policy_decision.plan = self._build_plan(
                policy_decision.intention, summary, context
            )
        if not policy_decision.perception_summary:
            policy_decision.perception_summary = dict(summary)
        if not policy_decision.summary:
            policy_decision.summary = (
                ", ".join(f"{k}:{v:.2f}" for k, v in summary.items())
                or "no-salient-modalities"
            )

        calibrated_confidence = self._calibrate_confidence(policy_decision.confidence)
        policy_decision.confidence = calibrated_confidence
        if policy_decision.focus is None and policy_decision.plan:
            policy_decision.focus = policy_decision.plan[0]

        tags = list(dict.fromkeys(policy_decision.tags)) if policy_decision.tags else []
        if policy_decision.focus and f"focus-{policy_decision.focus}" not in tags:
            tags.append(f"focus-{policy_decision.focus}")
        policy_decision.tags = tags

        self._remember(
            policy_decision.perception_summary,
            emotion,
            policy_decision.intention,
            policy_decision.confidence,
        )
        self._confidence_history.append(policy_decision.confidence)
        policy_decision.metadata.setdefault("confidence_calibrated", True)

        decision = {
            "intention": policy_decision.intention,
            "plan": list(policy_decision.plan),
            "confidence": policy_decision.confidence,
            "weights": dict(policy_decision.weights),
            "tags": list(policy_decision.tags),
            "focus": policy_decision.focus or policy_decision.intention,
            "summary": policy_decision.summary,
            "thought_trace": list(policy_decision.thought_trace),
            "perception_summary": dict(policy_decision.perception_summary),
            "policy_metadata": dict(policy_decision.metadata),
        }
        return decision



@dataclass
class WholeBrainSimulation:
    """Container object coordinating all brain subsystems."""

    visual: VisualCortex = field(default_factory=VisualCortex)
    auditory: AuditoryCortex = field(default_factory=AuditoryCortex)
    somatosensory: SomatosensoryCortex = field(default_factory=SomatosensoryCortex)
    cognition: CognitiveModule = field(default_factory=CognitiveModule)
    personality: PersonalityProfile = field(default_factory=PersonalityProfile)
    emotion: LimbicSystem = field(init=False)
    consciousness: ConsciousnessModel = field(default_factory=ConsciousnessModel)
    motor: MotorCortex = field(default_factory=MotorCortex)
    precision_motor: PrecisionMotorSystem = field(default_factory=PrecisionMotorSystem)
    curiosity: CuriosityState = field(default_factory=CuriosityState)
    cerebellum: Cerebellum = field(default_factory=Cerebellum)
    oscillations: NeuralOscillations = field(default_factory=NeuralOscillations)
    config: BrainRuntimeConfig = field(default_factory=BrainRuntimeConfig)
    self_learning: SelfLearningBrain = field(default_factory=SelfLearningBrain)
    perception_pipeline: SensoryPipeline = field(default_factory=SensoryPipeline)
    neuromorphic: bool = True
    neuromorphic_encoding: str = "rate"
    encoding_steps: int = 5
    encoding_time_scale: float = 1.0
    max_neurons: int = 128
    max_cache_size: int = 8
    cycle_index: int = field(init=False, default=0)
    last_perception: PerceptionSnapshot = field(init=False, default_factory=PerceptionSnapshot)
    last_context: Dict[str, Any] = field(init=False, default_factory=dict)
    last_learning_prediction: Dict[str, float] = field(init=False, default_factory=dict)
    last_decision: Dict[str, Any] = field(init=False, default_factory=dict)
    last_oscillation_state: Dict[str, float] = field(init=False, default_factory=dict)
    last_motor_result: Optional[NeuromorphicRunResult] = field(init=False, default=None)
    _spiking_cache: OrderedDict[tuple[int, str], NeuromorphicBackend] = field(init=False, default_factory=OrderedDict)
    _motor_backend: Optional[NeuromorphicBackend] = field(init=False, default=None)
    perception_history: deque[PerceptionSnapshot] = field(
        init=False, default_factory=lambda: deque(maxlen=32)
    )
    decision_history: deque[Dict[str, Any]] = field(
        init=False, default_factory=lambda: deque(maxlen=32)
    )
    telemetry_log: deque[Dict[str, Any]] = field(
        init=False, default_factory=lambda: deque(maxlen=64)
    )
    _stream_state: Dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.personality.clamp()
        self.emotion = LimbicSystem(self.personality)
        self.config.use_neuromorphic = self.neuromorphic
        self.motor.cerebellum = self.cerebellum
        self.motor.precision_system = self.precision_motor
        if hasattr(self.precision_motor, "basal_ganglia"):
            self.motor.basal_ganglia = self.precision_motor.basal_ganglia
        self.motor.spiking_backend = None
        self.last_oscillation_state = {}
        self.last_motor_result = None
        self._motor_backend = None
        self.last_perception = PerceptionSnapshot()
        self.last_context = {}
        self.last_learning_prediction = {}
        self.last_decision = {}
        self.cycle_index = 0
        history_size = max(4, self.max_cache_size)
        self.perception_history = deque(maxlen=history_size)
        self.decision_history = deque(maxlen=history_size)
        self.telemetry_log = deque(maxlen=max(8, history_size * 2))
        self._stream_state = {}

    @staticmethod
    def _perception_signature(snapshot: PerceptionSnapshot) -> str:
        parts = []
        for name, payload in sorted(snapshot.modalities.items()):
            spikes = payload.get("spike_counts") or []
            parts.append(f"{name}:{','.join(str(int(v)) for v in spikes)}")
        return "|".join(parts)

    def _estimate_novelty(self, snapshot: PerceptionSnapshot) -> float:
        if not snapshot.modalities:
            return 0.0
        novelty = 0.0
        denominator = 0.0
        for name, payload in snapshot.modalities.items():
            spikes = payload.get("spike_counts") or []
            denominator += len(spikes) or 1
            previous_snapshot = self.perception_history[-1] if self.perception_history else self.last_perception
            previous = previous_snapshot.modalities.get(name, {}) if previous_snapshot else {}
            previous_spikes = previous.get("spike_counts") or []
            length = max(len(spikes), len(previous_spikes))
            if length == 0:
                continue
            diff = 0.0
            for idx in range(length):
                current = float(spikes[idx]) if idx < len(spikes) else 0.0
                prior = float(previous_spikes[idx]) if idx < len(previous_spikes) else 0.0
                diff += abs(current - prior)
            novelty += diff / length
        if denominator <= 0:
            denominator = 1.0
        return max(0.0, min(1.0, novelty / denominator))

    def _ensure_motor_backend(self, neurons: int) -> NeuromorphicBackend:
        size = max(1, min(self.max_neurons, int(neurons) if neurons else 1))
        backend = self._motor_backend
        current_size = None
        if backend is not None:
            neuron_obj = getattr(getattr(backend, 'network', None), 'neurons', None)
            current_size = getattr(neuron_obj, 'size', None)
        if backend is None or current_size != size:
            cfg = SpikingNetworkConfig(n_neurons=size, idle_skip=True)
            backend = cfg.create_backend()
            self._motor_backend = backend
        backend.reset_state()
        self.motor.spiking_backend = backend
        return backend

    def _run_motor_neuromorphic(
        self,
        weights: Dict[str, float],
        intention: str,
        encoding_mode: str,
        modulators: Optional[Dict[str, float]] = None,
        oscillation_state: Optional[Dict[str, float]] = None,
    ) -> Optional[NeuromorphicRunResult]:
        if not self.neuromorphic or (not weights and not modulators):
            return None
        base_channels = ['observe', 'approach', 'withdraw', 'explore']
        extras = [key for key in weights.keys() if key not in base_channels]
        mod_channels: list[str] = []
        if modulators:
            mod_channels = [f"mod_{key}" for key in sorted(modulators.keys())]
        ordering = base_channels + extras + mod_channels
        vector = []
        for key in base_channels + extras:
            value = float(weights.get(key, 0.0))
            vector.append(max(0.0, min(1.0, value)))
        for channel_name in mod_channels:
            original = channel_name[4:]
            value = float(modulators.get(original, 0.0)) if modulators else 0.0
            vector.append(max(0.0, min(1.0, value)))
        backend = self._ensure_motor_backend(len(vector))
        encoder_kwargs: Dict[str, Any] = {}
        decoder_kwargs: Dict[str, Any] = {}
        mode = (encoding_mode or 'rate').lower()
        if mode == 'latency':
            encoder_kwargs['t_scale'] = self.encoding_time_scale
            decoder_kwargs['window'] = float(len(vector) or 1)
        else:
            steps = max(1, self.encoding_steps)
            encoder_kwargs['steps'] = steps
            decoder_kwargs['window'] = float(steps)
            mode = 'rate'
        metadata = {
            'intention': intention,
            'channels': ordering,
            'weights': dict(weights),
            'modulators': dict(modulators) if modulators else {},
        }
        if oscillation_state:
            metadata['oscillation'] = dict(oscillation_state)
        result = backend.run_sequence(
            [vector],
            encoding=mode,
            encoder_kwargs=encoder_kwargs,
            decoder='all',
            decoder_kwargs=decoder_kwargs,
            metadata=metadata,
            neuromodulation=oscillation_state,
            reset=True,
        )
        self.last_motor_result = result
        return result

    def _compute_oscillation_state(
        self,
        perception: PerceptionSnapshot,
        novelty: float,
    ) -> Dict[str, float]:
        if not self.config.metrics_enabled:
            return {}
        try:
            modalities = max(1, len(perception.modalities) or 1)
            num_osc = max(2, min(4, modalities))
            coupling = 0.4 + 0.6 * max(0.0, min(1.0, novelty))
            stimulus = 1.0 + max(0.0, min(1.0, self.curiosity.drive))
            criticality = 0.8 + 0.2 * max(0.0, min(1.0, self.personality.openness))
            waves = self.oscillations.generate_realistic_oscillations(
                num_oscillators=num_osc,
                duration=max(0.1, 0.1 * modalities),
                sample_rate=200,
                coupling_strength=coupling,
                stimulus=stimulus,
                criticality=criticality,
            )
            if getattr(waves, 'size', 0) == 0:
                return {}
            amplitude = float(np.mean(np.abs(waves)))
            amplitude_norm = float(np.tanh(amplitude))
            if getattr(waves, 'ndim', 0) >= 2 and waves.shape[0] > 1:
                correlation = np.corrcoef(waves)
                mask = ~np.eye(correlation.shape[0], dtype=bool)
                synchrony_index = float(np.mean(np.abs(correlation[mask]))) if mask.any() else 0.0
                modulation = float(np.mean(waves[-1]))
            else:
                synchrony_index = 0.0
                modulation = float(np.mean(waves))
            synchrony_norm = float(np.clip(synchrony_index, 0.0, 1.0))
            spectral = np.fft.rfft(waves, axis=-1)
            spectral_power = np.abs(spectral) ** 2
            mean_power = spectral_power.mean(axis=0)
            freqs = np.fft.rfftfreq(waves.shape[-1], d=1.0 / 200.0)
            if mean_power.size > 0:
                dominant_idx = int(np.argmax(mean_power))
                dominant_frequency = float(freqs[dominant_idx])
                rhythmicity = float(np.tanh(mean_power[dominant_idx]))
            else:
                dominant_frequency = 0.0
                rhythmicity = 0.0
            plasticity_gate = float(np.clip((amplitude_norm + synchrony_norm) * 0.5 + rhythmicity * 0.25, 0.0, 2.0))
            state = {
                'amplitude': amplitude,
                'amplitude_norm': amplitude_norm,
                'synchrony_index': synchrony_index,
                'synchrony_norm': synchrony_norm,
                'modulation': modulation,
                'coupling': coupling,
                'dominant_frequency': dominant_frequency,
                'rhythmicity': rhythmicity,
                'plasticity_gate': plasticity_gate,
            }
            self.last_oscillation_state = state
            return state
        except Exception as exc:
            logger.debug('Oscillation synthesis failed: %s', exc)
            return dict(self.last_oscillation_state)



    def _compose_thought_snapshot(
        self,
        decision: Dict[str, Any],
        memory_refs: List[dict[str, Any]],
    ) -> ThoughtSnapshot:
        plan_steps = list(decision.get("plan", []))
        summary = decision.get("summary") or (
            ', '.join(plan_steps) if plan_steps else decision.get("intention", "")
        )
        return ThoughtSnapshot(
            focus=str(decision.get("focus", decision.get("intention", "unknown"))),
            summary=summary,
            plan=plan_steps,
            confidence=float(decision.get("confidence", 0.5)),
            memory_refs=memory_refs[-3:],
            tags=list(decision.get("tags", [])),
        )

    def _compose_feeling_snapshot(
        self,
        emotion: EmotionSnapshot,
        oscillation_state: Dict[str, float],
        context_features: Dict[str, Any],
    ) -> FeelingSnapshot:
        descriptor = emotion.primary.value.lower()
        valence = float(emotion.dimensions.get("valence", emotion.mood))
        arousal = float(emotion.dimensions.get("arousal", abs(emotion.mood)))
        confidence = max(0.0, min(1.0, 1.0 - float(emotion.decay)))
        context_tags = {
            key
            for key, value in context_features.items()
            if isinstance(value, (int, float)) and value != 0
        }
        for key in oscillation_state:
            context_tags.add(f"osc_{key}")
        return FeelingSnapshot(
            descriptor=descriptor,
            valence=valence,
            arousal=arousal,
            mood=emotion.mood,
            confidence=confidence,
            context_tags=sorted(context_tags),
        )

    def process_cycle(self, input_data: Dict[str, Any]) -> BrainCycleResult:
        """Run a single perception-cognition-action cycle."""

        self.cycle_index += 1
        input_data = dict(input_data or {})
        use_neuromorphic = self.config.use_neuromorphic
        if use_neuromorphic != self.neuromorphic:
            use_neuromorphic = self.neuromorphic
            self.config.use_neuromorphic = self.neuromorphic

        if input_data.get("reset_streams"):
            self._stream_state.clear()
        drop_streams = input_data.get("drop_streams")
        if isinstance(drop_streams, Iterable) and not isinstance(drop_streams, (str, bytes)):
            for key in list(drop_streams):
                self._stream_state.pop(key, None)

        perception: Dict[str, Dict[str, Any]] = {}
        energy_used = 0.0
        idle_skipped = 0
        cycle_errors: List[str] = []
        cycle_telemetry: Dict[str, Any] = {
            "cycle_index": self.cycle_index,
            "use_neuromorphic": bool(use_neuromorphic),
        }

        def _resolve_signal(*keys: str) -> Any:
            for key in keys:
                if key in input_data:
                    return input_data[key]
            return None

        def _flatten_signal(value: Any) -> list[float]:
            if isinstance(value, Real):
                return [float(value)]
            if hasattr(value, "tolist"):
                return _flatten_signal(value.tolist())
            if isinstance(value, (list, tuple)):
                flat: list[float] = []
                for item in value:
                    flat.extend(_flatten_signal(item))
                return flat
            logger.debug("Unsupported sensory signal type: %s", type(value).__name__)
            return []

        def _consume_stream(modality: str, source: Any) -> Any:
            if source is None:
                return None
            try:
                if callable(source):
                    return source()
                if hasattr(source, "__next__"):
                    return next(source)
                if isinstance(source, (list, tuple, deque)):
                    return source[-1] if source else None
                if isinstance(source, dict):
                    for key in ("latest", "value", "data"):
                        if key in source:
                            return source[key]
                if isinstance(source, Iterable) and not isinstance(source, (str, bytes)):
                    iterator = iter(source)
                    return next(iterator)
            except StopIteration:
                return None
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug("Stream consumption failed for %s: %s", modality, exc)
                cycle_errors.append(f"stream:{modality}:{exc}")
                return None
            return source

        sensory_inputs: Dict[str, Any] = {}

        def _register_input(modality: str, value: Any, source: str) -> None:
            if value is None:
                return
            sensory_inputs[modality] = value
            modality_sources = cycle_telemetry.setdefault("modalities", {})
            modality_sources[modality] = source
            if input_data.get("persist_streams", True):
                self._stream_state[modality] = value

        streams = input_data.get("streams")
        if isinstance(streams, dict):
            for modality, source in streams.items():
                value = _consume_stream(modality, source)
                if value is not None:
                    _register_input(modality, value, "stream")

        stream_events = input_data.get("stream_events")
        if isinstance(stream_events, Iterable) and not isinstance(stream_events, (str, bytes)):
            for event in stream_events:
                try:
                    if not isinstance(event, dict):
                        continue
                    modality = event.get("modality") or event.get("name")
                    if not modality:
                        continue
                    value = event.get("value", event.get("data"))
                    if value is None:
                        continue
                    _register_input(modality, value, "event")
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.debug("Failed to process stream event %s: %s", event, exc)
                    cycle_errors.append(f"stream-event:{exc}")

        vision_signal = _resolve_signal("vision", "image")
        if vision_signal is not None:
            _register_input("vision", vision_signal, "direct")
        auditory_signal = _resolve_signal("auditory", "sound", "audio")
        if auditory_signal is not None:
            _register_input("auditory", auditory_signal, "direct")
        somatosensory_signal = _resolve_signal("somatosensory", "touch")
        if somatosensory_signal is not None:
            _register_input("somatosensory", somatosensory_signal, "direct")

        if input_data.get("use_cached_streams", True):
            for modality, cached_value in self._stream_state.items():
                sensory_inputs.setdefault(modality, cached_value)
                modality_sources = cycle_telemetry.setdefault("modalities", {})
                modality_sources.setdefault(modality, "cached")

        if use_neuromorphic:

            def _compress(vector: list[float], target: int | None = None) -> list[float]:
                if not vector:
                    return vector
                limit = self.max_neurons
                if target is not None:
                    limit = max(1, min(limit, int(target)))
                if len(vector) <= limit:
                    return vector
                chunk = max(1, math.ceil(len(vector) / limit))
                compressed: list[float] = []
                for i in range(0, len(vector), chunk):
                    segment = vector[i : i + chunk]
                    if segment:
                        compressed.append(sum(segment) / len(segment))
                    if len(compressed) == limit:
                        break
                return compressed or vector[:limit]

            def _select_bucket(length: int) -> int:
                if length <= 0:
                    return 0
                if length >= self.max_neurons:
                    return self.max_neurons
                power = 1 << (length - 1).bit_length()
                return min(self.max_neurons, power)

            def _encode_and_run(signal: Any, modality: str) -> None:
                nonlocal energy_used, idle_skipped
                try:
                    encoded = self.perception_pipeline.encode(modality, signal)
                except Exception as exc:
                    logger.debug("Perception encoding failed for %s: %s", modality, exc)
                    cycle_errors.append(f"encode:{modality}:{exc}")
                    encoded = EncodedSignal()
                source_vector = encoded.vector or _flatten_signal(signal)
                vector = _compress(source_vector)
                if not vector:
                    return
                base_signal = _flatten_signal(signal)
                bucket = _select_bucket(len(base_signal) or len(vector))
                if len(vector) > bucket:
                    vector = _compress(vector, target=bucket)
                if bucket == 0:
                    return
                if len(vector) < bucket:
                    vector = vector + [0.0] * (bucket - len(vector))

                encoding_mode = (self.neuromorphic_encoding or "rate").lower()
                cache_key = (bucket, encoding_mode)
                backend = self._spiking_cache.get(cache_key)
                if backend is None:
                    cfg = SpikingNetworkConfig(n_neurons=bucket, idle_skip=True)
                    backend = cfg.create_backend()
                    self._spiking_cache[cache_key] = backend
                else:
                    backend.reset_state()
                self._spiking_cache.move_to_end(cache_key)
                if len(self._spiking_cache) > self.max_cache_size:
                    self._spiking_cache.popitem(last=False)

                encoder_kwargs: Dict[str, Any] = {}
                decoder_kwargs: Dict[str, Any] = {}
                if encoding_mode == "latency":
                    encoder_kwargs["t_scale"] = self.encoding_time_scale
                    decoder_kwargs["window"] = float(len(vector) or 1)
                elif encoding_mode == "rate":
                    steps = max(1, self.encoding_steps)
                    encoder_kwargs["steps"] = steps
                    decoder_kwargs["window"] = float(steps)

                try:
                    result = backend.run_sequence(
                        [vector],
                        encoding=encoding_mode if encoding_mode in {"rate", "latency"} else None,
                        encoder_kwargs=encoder_kwargs,
                        decoder="all",
                        decoder_kwargs=decoder_kwargs,
                        metadata={"modality": modality},
                        reset=False,
                    )
                except Exception as exc:  # pragma: no cover - backend failure handling
                    logger.debug("Neuromorphic backend failed for %s: %s", modality, exc)
                    cycle_errors.append(f"spiking:{modality}:{exc}")
                    return

                raw_counts = list(result.spike_counts)
                entry: Dict[str, Any] = {
                    "spike_counts": raw_counts,
                    "spike_events": result.spike_events,
                    "average_rate": result.average_rate,
                    "vector": vector,
                    "metadata": {
                        "energy_used": result.energy_used,
                        "idle_skipped": result.idle_skipped,
                        "encoding": encoding_mode,
                    },
                }
                if encoded.features:
                    entry["features"] = encoded.features
                if encoded.metadata:
                    entry["metadata"].update(encoded.metadata)
                base_for_counts = list(base_signal) if base_signal else []
                if len(base_for_counts) < len(raw_counts):
                    base_for_counts.extend([0.0] * (len(raw_counts) - len(base_for_counts)))
                if base_for_counts and raw_counts:
                    max_value = max(base_for_counts)
                    min_value = min(base_for_counts)
                    if abs(max_value - min_value) <= 1e-9:
                        peak_index = base_for_counts.index(max_value)
                        normalised = [
                            1 if idx == peak_index else 0 for idx in range(len(raw_counts))
                        ]
                    else:
                        normalised = [
                            1 if abs(value - max_value) <= 1e-9 else 0
                            for value in base_for_counts[: len(raw_counts)]
                        ]
                    entry["metadata"]["raw_spike_counts"] = raw_counts
                    entry["spike_counts"] = normalised
                perception[modality] = entry
                energy_used += float(result.energy_used)
                idle_skipped += int(result.idle_skipped)

            for modality, signal in sensory_inputs.items():
                _encode_and_run(signal, modality)
        else:
            for modality, signal in sensory_inputs.items():
                try:
                    encoded = self.perception_pipeline.encode(modality, signal)
                except Exception as exc:
                    logger.debug("Perception encoding failed for %s: %s", modality, exc)
                    cycle_errors.append(f"encode:{modality}:{exc}")
                    encoded = EncodedSignal()
                vector = encoded.vector or _flatten_signal(signal)
                entry: Dict[str, Any] = {
                    "vector": vector,
                    "metadata": {"encoding": "analytic"},
                }
                if encoded.features:
                    entry["features"] = encoded.features
                if encoded.metadata:
                    entry["metadata"].update(encoded.metadata)
                perception[modality] = entry

        if "auditory" in perception and "audio" not in perception:
            perception["audio"] = dict(perception["auditory"])
        if "somatosensory" in perception and "touch" not in perception:
            perception["touch"] = dict(perception["somatosensory"])

        perception_snapshot = PerceptionSnapshot(modalities=dict(perception))
        novelty = self._estimate_novelty(perception_snapshot)
        if self.config.enable_curiosity_feedback:
            self.curiosity.update(novelty, self.personality)
        else:
            self.curiosity.decay()
            self.curiosity.last_novelty = novelty

        oscillation_state = self._compute_oscillation_state(perception_snapshot, novelty)

        raw_context = input_data.get("context", {})
        cognitive_context: Dict[str, Any] = dict(raw_context) if isinstance(raw_context, dict) else {}
        if "task" in input_data and "task" not in cognitive_context:
            cognitive_context["task"] = input_data["task"]
        if "is_salient" in input_data:
            cognitive_context["salience"] = bool(input_data.get("is_salient"))
        cognitive_context.setdefault("novelty", novelty)
        cognitive_context.setdefault("cycle_index", self.cycle_index)
        cognitive_context.setdefault("energy_used", float(energy_used))

        context_features: Dict[str, float] = {}
        for key, value in cognitive_context.items():
            if isinstance(value, Real):
                context_features[key] = float(value)
        context_features.setdefault("novelty", novelty)

        text_signal = _resolve_signal("text", "language", "stimulus", "narrative")
        text_stimulus = str(text_signal) if text_signal is not None else ""
        emotional_state = self.emotion.react(text_stimulus, context_features, self.config)
        emotion_snapshot = EmotionSnapshot(
            primary=emotional_state.emotion,
            intensity=float(emotional_state.intensity),
            mood=float(self.emotion.mood),
            dimensions=dict(emotional_state.dimensions),
            context=dict(emotional_state.context_weights),
            decay=float(emotional_state.decay),
            intent_bias=dict(emotional_state.intent_bias),
        )

        personality_snapshot = PersonalityProfile(
            openness=float(self.personality.openness),
            conscientiousness=float(self.personality.conscientiousness),
            extraversion=float(self.personality.extraversion),
            agreeableness=float(self.personality.agreeableness),
            neuroticism=float(self.personality.neuroticism),
        )

        learning_prediction: Dict[str, float] = {}
        reward_signal: float | None = None
        if self.config.enable_self_learning:
            signature = self._perception_signature(perception_snapshot) or f"cycle-{self.cycle_index}"
            usage = {
                "cpu": float(energy_used),
                "memory": float(
                    sum(
                        len(payload.get("spike_counts") or [])
                        for payload in perception_snapshot.modalities.values()
                    )
                ),
            }
            reward = emotional_state.dimensions.get("valence", 0.0) * emotional_state.intensity
            reward += context_features.get("safety", 0.0) * 0.1
            reward -= context_features.get("threat", 0.0) * 0.2
            sample = {
                "state": signature,
                "agent_id": str(input_data.get("agent_id", "whole_brain")),
                "usage": usage,
                "reward": max(-1.0, min(1.0, reward)),
            }
            reward_signal = sample["reward"]
            learning_prediction = self.self_learning.curiosity_driven_learning(sample) or {}

        decision = self.cognition.decide(
            perception_snapshot,
            emotion_snapshot,
            self.personality,
            self.curiosity,
            learning_prediction if learning_prediction else None,
            cognitive_context,
        )
        intention = decision["intention"]
        if oscillation_state:
            decision["oscillation_state"] = dict(oscillation_state)

        def _clamp(value: Any) -> float:
            try:
                return max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                return 0.0

        modulators: Dict[str, float] = {
            "novelty": _clamp(novelty),
            "curiosity": _clamp(self.curiosity.drive),
            "fatigue": _clamp(self.curiosity.fatigue),
            "confidence": _clamp(decision.get("confidence")),
            "valence": _clamp((emotion_snapshot.dimensions.get("valence", 0.0) + 1.0) / 2.0),
            "arousal": _clamp(emotion_snapshot.dimensions.get("arousal", 0.0)),
            "mood": _clamp((emotion_snapshot.mood + 1.0) / 2.0),
        }
        if "safety" in context_features:
            modulators["safety"] = _clamp(context_features["safety"])
        if "threat" in context_features:
            modulators["threat"] = _clamp(context_features["threat"])
        if self.config.enable_personality_modulation:
            modulators["openness"] = _clamp(self.personality.openness)
            modulators["conscientiousness"] = _clamp(self.personality.conscientiousness)
        if oscillation_state:
            for key, value in oscillation_state.items():
                if isinstance(value, (int, float)):
                    modulators[f"osc_{key}"] = _clamp(value)

        plan_parameters: Dict[str, Any] = {}
        if modulators:
            plan_parameters["modulators"] = dict(modulators)
        if oscillation_state:
            plan_parameters["oscillation"] = dict(oscillation_state)

        motor_result: Optional[NeuromorphicRunResult] = None
        if decision.get("weights"):
            motor_result = self._run_motor_neuromorphic(
                decision["weights"],
                intention,
                self.neuromorphic_encoding or "rate",
                modulators,
                oscillation_state,
            )
            if motor_result:
                decision["motor_channels"] = list(motor_result.metadata.get("channels", []))
                decision["motor_spike_counts"] = list(motor_result.spike_counts)
        elif self.last_motor_result is not None:
            self.last_motor_result = None

        if decision.get("weights"):
            plan_parameters["weights"] = {
                key: float(value) for key, value in decision["weights"].items()
            }
        balance_hint = self.cerebellum.balance_control(f"novelty:{novelty:.3f}")
        plan_parameters["cerebellum_hint"] = balance_hint
        if motor_result:
            plan_parameters["neuromorphic_result"] = motor_result
            plan_parameters["motor_channels"] = list(motor_result.metadata.get("channels", []))
            if motor_result.average_rate:
                plan_parameters["motor_rate"] = list(motor_result.average_rate)

        try:
            plan = self.motor.plan_movement(intention, parameters=plan_parameters)
        except Exception as exc:  # pragma: no cover - defensive planning fallback
            logger.debug("Motor planning failed: %s", exc)
            cycle_errors.append(f"motor-plan:{exc}")
            plan = MotorPlan(
                intention=intention,
                stages=[f"fallback_{intention}"],
                parameters=dict(plan_parameters),
                metadata={
                    "plan_summary": f"fallback_{intention}",
                    "fallback": True,
                    "error": str(exc),
                },
            )
        if motor_result:
            plan.metadata["neuromorphic"] = motor_result.to_dict()
        try:
            action = self.motor.execute_action(plan)
        except Exception as exc:  # pragma: no cover - defensive execution fallback
            logger.debug("Motor execution failed: %s", exc)
            cycle_errors.append(f"motor-execute:{exc}")
            action = MotorExecutionResult(False, str(exc), telemetry={}, error=str(exc))

        external_feedback = None
        for key in ("motor_feedback", "execution_feedback", "actuator_feedback", "sensor_feedback"):
            if key in input_data:
                external_feedback = input_data[key]
                break
        if external_feedback is not None:
            try:
                self.motor.train(external_feedback)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.debug("Motor training from external feedback failed: %s", exc)

        execution_feedback = action if isinstance(action, MotorExecutionResult) else None
        execution_metrics = self.motor.parse_feedback_metrics(execution_feedback, base_reward=reward_signal)
        external_metrics = (
            self.motor.parse_feedback_metrics(external_feedback, base_reward=reward_signal)
            if external_feedback is not None
            else {}
        )
        feedback_metrics: Dict[str, float] = dict(execution_metrics)
        for key, value in external_metrics.items():
            if key in feedback_metrics and feedback_metrics[key] not in {0.0}:
                feedback_metrics[key] = (feedback_metrics[key] + value) / 2.0
            else:
                feedback_metrics[key] = value
        if reward_signal is not None and "reward" not in feedback_metrics:
            feedback_metrics["reward"] = float(reward_signal)
        if feedback_metrics:
            plan_parameters.setdefault("feedback_metrics", dict(feedback_metrics))
            plan.metadata["feedback_metrics"] = dict(feedback_metrics)
            decision["feedback_metrics"] = dict(feedback_metrics)

        curiosity_snapshot = CuriosityState(
            drive=self.curiosity.drive,
            novelty_preference=self.curiosity.novelty_preference,
            fatigue=self.curiosity.fatigue,
            last_novelty=self.curiosity.last_novelty,
        )

        intent = CognitiveIntent(
            intention=intention,
            salience=bool(input_data.get("is_salient", False)),
            plan=list(decision.get("plan", [])),
            confidence=float(decision.get("confidence", 0.0)),
            weights=dict(decision.get("weights", {})),
            tags=list(decision.get("tags", [])),
        )

        thought_snapshot = self._compose_thought_snapshot(
            decision,
            self.cognition.recall(),
        )
        feeling_snapshot = self._compose_feeling_snapshot(
            emotion_snapshot,
            oscillation_state,
            context_features,
        )

        metrics: Dict[str, float] = {}
        if self.config.metrics_enabled:
            metrics.update({"modalities": float(len(perception_snapshot.modalities))})
            metrics.update(
                {
                    "novelty_signal": novelty,
                    "energy_used": float(energy_used),
                    "idle_skipped": float(idle_skipped),
                    "cycle_index": float(self.cycle_index),
                }
            )
            metrics.update(self.curiosity.as_metrics())
            metrics.update(emotion_snapshot.as_metrics())
            if oscillation_state:
                metrics.update({f"osc_{k}": float(v) for k, v in oscillation_state.items()})
            if motor_result:
                metrics["motor_energy"] = float(motor_result.energy_used)
                metrics["motor_idle_skipped"] = float(motor_result.idle_skipped)
                if motor_result.spike_counts:
                    metrics["motor_spike_avg"] = float(
                        sum(motor_result.spike_counts) / len(motor_result.spike_counts)
                    )
                if motor_result.average_rate:
                    metrics["motor_rate_avg"] = float(
                        sum(motor_result.average_rate) / len(motor_result.average_rate)
                    )
            if feedback_metrics:
                metrics.update({f"feedback_{k}": float(v) for k, v in feedback_metrics.items()})
            intent_metrics = intent.as_metrics()
            metrics["intent_confidence"] = intent_metrics.get(
                "intent_confidence", intent.confidence
            )
            weights = decision.get("weights", {})
            for key in ("approach", "withdraw", "explore", "observe"):
                metrics[f"strategy_bias_{key}"] = float(weights.get(key, 0.0))
            if self.config.enable_plan_logging:
                metrics.update({k: v for k, v in intent_metrics.items() if k != "intent_confidence"})
                metrics["plan_length"] = float(len(decision.get("plan", [])))
            if learning_prediction:
                metrics.update(
                    {f"learning_{k}": float(v) for k, v in learning_prediction.items()}
                )
            if cycle_errors:
                metrics["cycle_error_count"] = float(len(cycle_errors))

        policy_metadata = dict(decision.get("policy_metadata", {}))
        unique_errors = list(dict.fromkeys(cycle_errors)) if cycle_errors else []
        if unique_errors:
            decision["errors"] = list(unique_errors)

        energy_used_int = int(round(energy_used))
        result = BrainCycleResult(
            perception=perception_snapshot,
            emotion=emotion_snapshot,
            intent=intent,
            personality=personality_snapshot,
            curiosity=curiosity_snapshot,
            energy_used=energy_used_int,
            idle_skipped=int(idle_skipped),
            thoughts=thought_snapshot,
            feeling=feeling_snapshot,
            metrics=metrics,
            metadata={
                "plan": plan.describe(),
                "executed_action": str(action),
                "cognitive_plan": ",".join(decision.get("plan", []))
                if self.config.enable_plan_logging
                else None,
                "memory_size": str(len(self.cognition.episodic_memory)),
                "context_task": str(cognitive_context.get("task"))
                if cognitive_context.get("task") is not None
                else None,
                "oscillation_state": str(oscillation_state) if oscillation_state else None,
                "motor_spike_counts": str(motor_result.spike_counts)
                if motor_result
                else None,
                "motor_average_rate": str(motor_result.average_rate)
                if motor_result and motor_result.average_rate
                else None,
                "motor_energy": str(motor_result.energy_used) if motor_result else None,
                "feedback_metrics": dict(feedback_metrics) if feedback_metrics else None,
                "policy": policy_metadata.get("policy"),
                "policy_metadata": policy_metadata or None,
                "cycle_errors": unique_errors or None,
            },
        )
        self.last_perception = perception_snapshot
        self.last_context = cognitive_context
        self.last_learning_prediction = learning_prediction
        self.last_decision = decision
        self.perception_history.append(perception_snapshot)
        self.decision_history.append(dict(decision))
        cycle_telemetry.update(
            {
                "intention": intention,
                "confidence": decision.get("confidence"),
                "policy": policy_metadata.get("policy"),
            }
        )
        if unique_errors:
            cycle_telemetry["errors"] = unique_errors
        if plan:
            cycle_telemetry["plan_length"] = len(plan.stages)
        if decision.get("plan"):
            cycle_telemetry["cognitive_plan"] = list(decision.get("plan", []))
        self.telemetry_log.append(dict(cycle_telemetry))
        return result

    def update_config(self, config: BrainRuntimeConfig) -> None:
        """Replace runtime configuration and keep derived flags in sync."""

        self.config = config
        self.neuromorphic = config.use_neuromorphic

    def get_decision_trace(self, limit: int = 5) -> List[dict[str, Any]]:
        """Return recent cognitive decisions for inspection."""

        return self.cognition.recall(limit)

    def get_strategy_modulation(self) -> Dict[str, float]:
        """Expose the latest action weights for agent loop adjustments."""

        weights = self.last_decision.get("weights", {}) if isinstance(self.last_decision, dict) else {}
        return {
            "approach": float(weights.get("approach", 0.0)),
            "withdraw": float(weights.get("withdraw", 0.0)),
            "explore": float(weights.get("explore", 0.0)),
            "observe": float(weights.get("observe", 0.0)),
            "curiosity_drive": float(self.curiosity.drive),
        }

__all__ = ["WholeBrainSimulation"]
