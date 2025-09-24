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

import hashlib
import logging
import math
import numpy as np
from dataclasses import dataclass, field
from numbers import Real
from collections import OrderedDict, deque
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
        history: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> CognitiveDecision:
        raise NotImplementedError


class HeuristicCognitivePolicy(CognitivePolicy):
    """Legacy heuristic policy retained for deterministic fallbacks."""

    def __init__(self) -> None:
        self.planner = StructuredPlanner()

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
        history: Optional[Sequence[Dict[str, Any]]] = None,
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
        try:
            plan = self.planner.generate(
                intention,
                focus,
                context,
                perception,
                summary,
                emotion,
                curiosity,
                history,
                learning_prediction,
            )
        except Exception:
            plan = []
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


class StructuredPlanner:
    """Probabilistic planner assembling production-grade cognitive plans."""

    name = "structured"

    class _PlanTransitionModel:
        def __init__(
            self,
            intention: str,
            vocabulary: Sequence[str],
            prototype: Sequence[str],
            rng: np.random.Generator,
            min_steps: int,
        ) -> None:
            self.intention = intention
            self.vocabulary = tuple(dict.fromkeys(vocabulary))
            self.prototype = list(prototype)
            self._rng = rng
            self.min_steps = min_steps
            self.state_tokens = ("<START>",) + self.vocabulary
            self.target_tokens = self.vocabulary + ("<END>",)
            self.state_index = {token: index for index, token in enumerate(self.state_tokens)}
            self.target_index = {token: index for index, token in enumerate(self.target_tokens)}
            self._feature_mean: np.ndarray | None = None
            self._feature_std: np.ndarray | None = None
            self._weights: np.ndarray | None = None

        def train(self, samples: Sequence[Tuple[np.ndarray, Sequence[str]]]) -> None:
            if not samples:
                raise ValueError("Planner requires training samples")
            state_dim = len(self.state_tokens)
            base_features: List[np.ndarray] = []
            state_vectors: List[np.ndarray] = []
            targets: List[int] = []
            for base, plan in samples:
                sequence = list(plan) or list(self.prototype)
                states = ["<START>", *sequence]
                next_steps = [*sequence, "<END>"]
                for state, step in zip(states, next_steps):
                    base_features.append(np.asarray(base, dtype=np.float64))
                    state_vector = np.zeros(state_dim, dtype=np.float64)
                    state_vector[self.state_index[state]] = 1.0
                    state_vectors.append(state_vector)
                    targets.append(self.target_index[step])
            base_matrix = np.vstack(base_features)
            state_matrix = np.vstack(state_vectors)
            mean = base_matrix[:, 1:].mean(axis=0)
            std = base_matrix[:, 1:].std(axis=0)
            std[std < 1e-6] = 1.0
            self._feature_mean = mean
            self._feature_std = std
            norm_base = base_matrix.copy()
            norm_base[:, 1:] = (norm_base[:, 1:] - mean) / std
            inputs = np.hstack([norm_base, state_matrix])
            n_samples = inputs.shape[0]
            n_targets = len(self.target_tokens)
            weights = self._rng.normal(0.0, 0.06, size=(n_targets, inputs.shape[1]))
            one_hot = np.zeros((n_samples, n_targets), dtype=np.float64)
            one_hot[np.arange(n_samples), targets] = 1.0
            learning_rate = 0.05
            regularisation = 0.002
            for _ in range(320):
                logits = inputs @ weights.T
                logits -= logits.max(axis=1, keepdims=True)
                exp_logits = np.exp(logits)
                probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
                grad = (probs - one_hot).T @ inputs / n_samples
                weights -= learning_rate * (grad + regularisation * weights)
            self._weights = weights

        def _prepare(self, base: np.ndarray) -> np.ndarray:
            arr = np.asarray(base, dtype=np.float64)
            if arr.ndim != 1:
                arr = arr.reshape(-1)
            if self._feature_mean is not None and self._feature_std is not None:
                normalised = arr.copy()
                normalised[1:] = (normalised[1:] - self._feature_mean) / (self._feature_std + 1e-6)
                return normalised
            return arr

        def ranked_steps(self, base: np.ndarray, state: str) -> List[str]:
            if self._weights is None:
                raise RuntimeError("Planner model not trained")
            prepared = self._prepare(base)
            state_vector = np.zeros(len(self.state_tokens), dtype=np.float64)
            state_vector[self.state_index.get(state, 0)] = 1.0
            inputs = np.concatenate([prepared, state_vector])
            logits = self._weights @ inputs
            logits -= logits.max()
            exp_logits = np.exp(logits)
            denom = exp_logits.sum()
            if denom <= 0.0:
                probs = np.full_like(exp_logits, 1.0 / exp_logits.size)
            else:
                probs = exp_logits / denom
            ranking = np.argsort(probs)[::-1]
            return [self.target_tokens[index] for index in ranking]

        def finalise(self, plan: Sequence[str], min_steps: int) -> List[str]:
            dedup: List[str] = []
            seen: set[str] = set()
            for step in plan:
                if step and step != "<END>" and step not in seen:
                    dedup.append(step)
                    seen.add(step)
            for fallback in self.prototype:
                if len(dedup) >= min_steps:
                    break
                if fallback not in seen:
                    dedup.append(fallback)
                    seen.add(fallback)
            if len(dedup) < min_steps:
                needed = min_steps - len(dedup)
                dedup.extend(["log_cognitive_trace"] * needed)
            if "archive_cognitive_trace" not in seen:
                dedup.append("archive_cognitive_trace")
            return dedup

    def __init__(self, min_steps: int = 3, *, rng: np.random.Generator | None = None) -> None:
        self.min_steps = max(1, int(min_steps))
        self._rng = rng or np.random.default_rng(41)
        self._models = self._train_models(self._default_corpus())

    @staticmethod
    def _focus_signature(focus: Optional[str]) -> List[float]:
        if not focus:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        focus_lower = focus.lower()
        digest = hashlib.sha256(focus_lower.encode("utf-8")).digest()
        return [
            1.0,
            min(len(focus_lower) / 16.0, 1.0),
            digest[0] / 255.0,
            digest[1] / 255.0,
            float("vision" in focus_lower or "visual" in focus_lower),
            float("audio" in focus_lower or "auditory" in focus_lower or "sound" in focus_lower),
            float("touch" in focus_lower or "somato" in focus_lower or "haptic" in focus_lower),
        ]

    @staticmethod
    def _emotion_features(emotion: EmotionSnapshot | Dict[str, float]) -> Tuple[float, float, float, float]:
        if isinstance(emotion, EmotionSnapshot):
            valence = float(emotion.dimensions.get("valence", 0.0))
            arousal = float(emotion.dimensions.get("arousal", emotion.intensity))
            dominance = float(emotion.dimensions.get("dominance", 0.0))
            intensity = float(emotion.intensity)
        else:
            valence = float(emotion.get("valence", 0.0))
            arousal = float(emotion.get("arousal", emotion.get("intensity", 0.35)))
            dominance = float(emotion.get("dominance", 0.0))
            intensity = float(emotion.get("intensity", 0.4))
        return valence, arousal, dominance, intensity

    @staticmethod
    def _curiosity_features(curiosity: CuriosityState | Dict[str, float]) -> Tuple[float, float, float, float]:
        if isinstance(curiosity, CuriosityState):
            return (
                float(curiosity.drive),
                float(curiosity.novelty_preference),
                float(curiosity.fatigue),
                float(curiosity.last_novelty),
            )
        return (
            float(curiosity.get("drive", 0.3)),
            float(curiosity.get("novelty_preference", curiosity.get("novelty", 0.3))),
            float(curiosity.get("fatigue", 0.2)),
            float(curiosity.get("last_novelty", 0.3)),
        )

    def _build_feature_vector(
        self,
        intention: str,
        context: Mapping[str, Any],
        summary: Mapping[str, float],
        emotion: EmotionSnapshot | Dict[str, float],
        curiosity: CuriosityState | Dict[str, float],
        learning: Optional[Mapping[str, float]] = None,
        focus: Optional[str] = None,
        history: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> np.ndarray:
        vector: List[float] = [1.0]
        context_keys = ("safety", "threat", "novelty", "social", "fatigue", "support")
        for key in context_keys:
            vector.append(float(context.get(key, 0.0)))
        valence, arousal, dominance, intensity = self._emotion_features(emotion)
        vector.extend([valence, arousal, dominance, intensity])
        drive, novelty_preference, fatigue, last_novelty = self._curiosity_features(curiosity)
        vector.extend([drive, novelty_preference, fatigue, last_novelty])
        summary_values = np.asarray(list(summary.values()), dtype=np.float64)
        if summary_values.size:
            total = float(summary_values.sum()) or 1.0
            normalised = summary_values / total
            entropy = float(-np.sum(normalised * np.log(normalised + 1e-6)))
            max_value = float(normalised.max())
        else:
            entropy = 0.0
            max_value = 0.0
        vector.extend(
            [
                max_value,
                float(summary.get("vision", 0.0)),
                float(summary.get("auditory", 0.0)),
                float(summary.get("somatosensory", 0.0)),
                float(summary.get("proprioception", 0.0)),
                entropy,
                min(1.0, len(summary) / 6.0),
            ]
        )
        learning_data = learning or {}
        vector.extend(
            [
                float(learning_data.get("cpu", 0.0)),
                float(learning_data.get("memory", 0.0)),
            ]
        )
        history_list = list(history or [])
        if history_list:
            same = sum(
                1
                for item in history_list
                if str(item.get("intention", "")).lower() == intention
            )
            confidences = [float(item.get("confidence", 0.0)) for item in history_list]
            mean_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0
        else:
            same = 0
            mean_conf = 0.0
        vector.extend([min(1.0, same / 4.0), mean_conf])
        vector.extend(self._focus_signature(focus))
        return np.asarray(vector, dtype=np.float64)

    def _train_models(self, corpus: Sequence[Dict[str, Any]]) -> Dict[str, "StructuredPlanner._PlanTransitionModel"]:
        grouped: Dict[str, List[Tuple[np.ndarray, Sequence[str]]]] = {}
        vocabulary: Dict[str, List[str]] = {}
        prototypes: Dict[str, List[str]] = {}
        for sample in corpus:
            intention = sample["intention"].lower()
            plan = list(sample["plan"])
            focus = sample.get("focus")
            features = self._build_feature_vector(
                intention,
                sample.get("context", {}),
                sample.get("summary", {}),
                sample.get("emotion", {}),
                sample.get("curiosity", {}),
                sample.get("learning", {}),
                focus,
                sample.get("history"),
            )
            grouped.setdefault(intention, []).append((features, plan))
            vocabulary.setdefault(intention, []).extend(plan)
            if intention not in prototypes or len(plan) > len(prototypes[intention]):
                prototypes[intention] = plan
        models: Dict[str, StructuredPlanner._PlanTransitionModel] = {}
        for intention, samples in grouped.items():
            model = StructuredPlanner._PlanTransitionModel(
                intention,
                vocabulary[intention],
                prototypes[intention],
                self._rng,
                self.min_steps,
            )
            model.train(samples)
            models[intention] = model
        return models

    def _default_corpus(self) -> List[Dict[str, Any]]:
        return [
            {
                "intention": "observe",
                "plan": [
                    "stabilise_attention",
                    "collect_multimodal_snapshot",
                    "analyse_vision_salience",
                    "update_world_model",
                    "broadcast_situation_report",
                ],
                "focus": "vision",
                "context": {"safety": 0.6, "novelty": 0.3, "social": 0.4},
                "summary": {"vision": 0.5, "auditory": 0.3, "somatosensory": 0.2},
                "emotion": {"valence": 0.25, "arousal": 0.35, "dominance": 0.1, "intensity": 0.4},
                "curiosity": {"drive": 0.35, "novelty_preference": 0.32, "fatigue": 0.25, "last_novelty": 0.28},
                "learning": {"cpu": 0.6, "memory": 0.5},
                "history": [{"intention": "observe", "confidence": 0.55}],
            },
            {
                "intention": "observe",
                "plan": [
                    "stabilise_attention",
                    "collect_multimodal_snapshot",
                    "analyse_auditory_salience",
                    "update_world_model",
                    "broadcast_situation_report",
                ],
                "focus": "auditory",
                "context": {"safety": 0.55, "novelty": 0.25},
                "summary": {"vision": 0.3, "auditory": 0.5, "somatosensory": 0.2},
                "emotion": {"valence": 0.2, "arousal": 0.3, "dominance": 0.05, "intensity": 0.35},
                "curiosity": {"drive": 0.4, "novelty_preference": 0.3, "fatigue": 0.22, "last_novelty": 0.26},
                "learning": {"cpu": 0.55, "memory": 0.45},
            },
            {
                "intention": "approach",
                "plan": [
                    "identify_positive_target",
                    "compute_safe_path",
                    "coordinate_motor_system",
                    "engage_target",
                    "record_feedback",
                ],
                "focus": "vision",
                "context": {"safety": 0.7, "social": 0.65},
                "summary": {"vision": 0.55, "auditory": 0.25, "somatosensory": 0.2},
                "emotion": {"valence": 0.6, "arousal": 0.45, "dominance": 0.2, "intensity": 0.55},
                "curiosity": {"drive": 0.6, "novelty_preference": 0.5, "fatigue": 0.18, "last_novelty": 0.4},
                "learning": {"cpu": 0.35, "memory": 0.3},
                "history": [{"intention": "approach", "confidence": 0.5}],
            },
            {
                "intention": "approach",
                "plan": [
                    "identify_positive_target",
                    "compute_safe_path",
                    "establish_social_contact",
                    "engage_target",
                    "record_feedback",
                ],
                "focus": "social",
                "context": {"safety": 0.68, "social": 0.75},
                "summary": {"vision": 0.45, "auditory": 0.35, "somatosensory": 0.2},
                "emotion": {"valence": 0.65, "arousal": 0.5, "dominance": 0.22, "intensity": 0.6},
                "curiosity": {"drive": 0.55, "novelty_preference": 0.45, "fatigue": 0.2, "last_novelty": 0.38},
                "learning": {"cpu": 0.4, "memory": 0.35},
            },
            {
                "intention": "withdraw",
                "plan": [
                    "elevate_alert_state",
                    "assess_risk_vectors",
                    "select_evasive_route",
                    "reinforce_safety_perimeter",
                    "log_retreat_outcome",
                ],
                "focus": "threat",
                "context": {"threat": 0.85, "safety": 0.2, "support": 0.3},
                "summary": {"vision": 0.3, "auditory": 0.4, "somatosensory": 0.3},
                "emotion": {"valence": -0.55, "arousal": 0.65, "dominance": -0.25, "intensity": 0.58},
                "curiosity": {"drive": 0.3, "novelty_preference": 0.25, "fatigue": 0.4, "last_novelty": 0.2},
                "learning": {"cpu": 0.6, "memory": 0.55},
                "history": [{"intention": "withdraw", "confidence": 0.55}],
            },
            {
                "intention": "withdraw",
                "plan": [
                    "elevate_alert_state",
                    "assess_risk_vectors",
                    "notify_support_channel",
                    "reinforce_safety_perimeter",
                    "log_retreat_outcome",
                ],
                "focus": "support",
                "context": {"threat": 0.75, "support": 0.6},
                "summary": {"vision": 0.25, "auditory": 0.45, "somatosensory": 0.3},
                "emotion": {"valence": -0.45, "arousal": 0.6, "dominance": -0.2, "intensity": 0.5},
                "curiosity": {"drive": 0.32, "novelty_preference": 0.28, "fatigue": 0.35, "last_novelty": 0.22},
                "learning": {"cpu": 0.55, "memory": 0.5},
            },
            {
                "intention": "explore",
                "plan": [
                    "probe_salient_focus",
                    "map_unexplored_regions",
                    "sample_novel_patterns",
                    "synthesise_novelty_brief",
                ],
                "focus": "novelty",
                "context": {"novelty": 0.82, "safety": 0.5},
                "summary": {"vision": 0.35, "auditory": 0.3, "somatosensory": 0.35},
                "emotion": {"valence": 0.4, "arousal": 0.55, "dominance": 0.1, "intensity": 0.52},
                "curiosity": {"drive": 0.75, "novelty_preference": 0.82, "fatigue": 0.18, "last_novelty": 0.7},
                "learning": {"cpu": 0.4, "memory": 0.35},
                "history": [{"intention": "explore", "confidence": 0.5}],
            },
            {
                "intention": "explore",
                "plan": [
                    "discover_salient_focus",
                    "map_unexplored_regions",
                    "predict_exploration_value",
                    "synthesise_novelty_brief",
                ],
                "focus": "vision",
                "context": {"novelty": 0.7, "safety": 0.45},
                "summary": {"vision": 0.4, "auditory": 0.25, "somatosensory": 0.35},
                "emotion": {"valence": 0.35, "arousal": 0.5, "dominance": 0.05, "intensity": 0.48},
                "curiosity": {"drive": 0.7, "novelty_preference": 0.75, "fatigue": 0.2, "last_novelty": 0.68},
                "learning": {"cpu": 0.38, "memory": 0.32},
            },
        ]

    def generate(
        self,
        intention: str,
        focus: Optional[str],
        context: Dict[str, Any],
        perception: PerceptionSnapshot,
        summary: Dict[str, float],
        emotion: EmotionSnapshot,
        curiosity: CuriosityState,
        history: Optional[Sequence[Dict[str, Any]]] = None,
        learning_prediction: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        intention_key = (intention or "observe").lower()
        model = self._models.get(intention_key)
        if model is None:
            intention_key = next(iter(self._models))
            model = self._models[intention_key]
        focus_token = focus or (max(summary, key=summary.get) if summary else context.get("target"))
        features = self._build_feature_vector(
            intention_key,
            context,
            summary,
            emotion,
            curiosity,
            learning_prediction,
            focus_token,
            history,
        )
        assembled: List[str] = []
        state = "<START>"
        max_steps = max(self.min_steps + 2, len(model.prototype) + 1)
        for _ in range(max_steps):
            ranked = model.ranked_steps(features, state)
            chosen: Optional[str] = None
            for candidate in ranked:
                if candidate == "<END>":
                    if len(assembled) >= self.min_steps:
                        state = "<END>"
                        break
                    continue
                if candidate in assembled and len(assembled) < len(model.prototype):
                    continue
                chosen = candidate
                break
            if state == "<END>" or chosen is None:
                break
            assembled.append(chosen)
            state = chosen
        return model.finalise(assembled, self.min_steps)


class AdaptivePlanner:
    """Graph-based planner combining structured plans with contextual search."""

    def __init__(self, base: StructuredPlanner | None = None, min_steps: int = 4) -> None:
        self.base = base or StructuredPlanner(min_steps=min_steps)
        self.min_steps = max(3, int(min_steps))
        self._graph = self._build_graph()

    @staticmethod
    def _build_graph() -> Dict[str, List[Tuple[str, Tuple[str, ...]]]]:
        def edges(*pairs: Tuple[str, Tuple[str, ...]]):
            return list(pairs)

        return {
            "observe": edges(
                ("stabilise_attention", ("collect_multimodal_snapshot",)),
                ("collect_multimodal_snapshot", ("analyse_{focus}_salience", "update_world_model")),
                ("update_world_model", ("broadcast_situation_report",)),
                ("broadcast_situation_report", ()),
            ),
            "approach": edges(
                ("validate_positive_target", ("calculate_approach_vector",)),
                ("calculate_approach_vector", ("coordinate_motor_system", "predict_contact_outcome")),
                ("coordinate_motor_system", ("engage_target", "record_feedback")),
                ("engage_target", ("record_feedback",)),
                ("record_feedback", ()),
            ),
            "withdraw": edges(
                ("elevate_alert_state", ("select_evasive_route", "notify_support_channel")),
                ("select_evasive_route", ("reinforce_safety_perimeter", "update_world_model")),
                ("notify_support_channel", ("reinforce_safety_perimeter",)),
                ("reinforce_safety_perimeter", ("log_retreat_outcome",)),
                ("log_retreat_outcome", ()),
            ),
            "explore": edges(
                ("sample_novel_patterns", ("map_unexplored_regions", "log_novelty_metrics")),
                ("map_unexplored_regions", ("predict_exploration_value",)),
                ("log_novelty_metrics", ("synthesise_novelty_brief",)),
                ("predict_exploration_value", ("expand_search_radius", "synthesise_novelty_brief")),
                ("expand_search_radius", ()),
                ("synthesise_novelty_brief", ()),
            ),
        }

    def _render_step(self, template: str, focus: Optional[str], context: Dict[str, Any]) -> str:
        focus_token = focus or context.get("target") or "salience"
        rendered = template.replace("{focus}", str(focus_token))
        if "{threat_level}" in rendered:
            rendered = rendered.replace(
                "{threat_level}",
                f"{float(context.get('threat', 0.0)):.2f}",
            )
        return rendered

    def _augment_with_context(
        self,
        intention: str,
        base_plan: List[str],
        context: Dict[str, Any],
        curiosity: CuriosityState,
        learning_prediction: Optional[Dict[str, float]],
    ) -> List[str]:
        enriched = list(base_plan)
        threat = float(context.get("threat", 0.0))
        safety = float(context.get("safety", 0.0))
        novelty = float(context.get("novelty", curiosity.last_novelty))
        fatigue = float(context.get("fatigue", curiosity.fatigue))
        if intention == "withdraw" and threat > 0.7:
            enriched.append("deploy_countermeasures")
        if intention == "approach" and safety > 0.6:
            enriched.append("capture_positive_feedback")
        if intention == "explore" and novelty > 0.6:
            enriched.append("prioritise_unmapped_regions")
        if fatigue > 0.5:
            enriched.append("schedule_recovery_cycle")
        if learning_prediction:
            cpu = float(learning_prediction.get("cpu", 0.0))
            if cpu > 0.7:
                enriched.append("rebalance_cognitive_load")
        return list(dict.fromkeys(enriched))

    def _search_plan(
        self,
        intention: str,
        focus: Optional[str],
        context: Dict[str, Any],
        curiosity: CuriosityState,
        learning_prediction: Optional[Dict[str, float]],
    ) -> List[str]:
        queue: deque[Tuple[str, int]] = deque([(intention, 0)])
        visited_states: set[str] = set()
        plan: List[str] = []
        while queue and len(plan) < self.min_steps + 2:
            state, depth = queue.popleft()
            if state in visited_states:
                continue
            visited_states.add(state)
            edges = self._graph.get(state, [])
            if not edges and state not in self._graph:
                rendered = self._render_step(state, focus, context)
                if rendered not in plan:
                    plan.append(rendered)
                continue
            for template, successors in edges:
                rendered = self._render_step(template, focus, context)
                if rendered not in plan:
                    plan.append(rendered)
                for successor in successors:
                    queue.append((successor, depth + 1))
        if len(plan) < self.min_steps:
            plan.append("log_cognitive_trace")
        return self._augment_with_context(intention, plan, context, curiosity, learning_prediction)

    def generate(
        self,
        intention: str,
        focus: Optional[str],
        context: Dict[str, Any],
        perception: PerceptionSnapshot,
        summary: Dict[str, float],
        emotion: EmotionSnapshot,
        curiosity: CuriosityState,
        history: Optional[Sequence[Dict[str, Any]]] = None,
        learning_prediction: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        try:
            base_plan = self.base.generate(
                intention,
                focus,
                context,
                perception,
                summary,
                emotion,
                curiosity,
                history,
                learning_prediction,
            )
        except Exception:
            base_plan = []
        if base_plan:
            enriched = self._augment_with_context(
                intention, base_plan, context, curiosity, learning_prediction
            )
            if len(enriched) >= self.min_steps:
                return enriched
        return self._search_plan(
            intention,
            focus,
            context,
            curiosity,
            learning_prediction,
        )


class ProductionCognitivePolicy(CognitivePolicy):
    """Softmax policy trained via regularised multi-class regression."""

    INTENTIONS: Tuple[str, ...] = ("observe", "approach", "withdraw", "explore")
    TRAINING_EPOCHS = 240

    def __init__(
        self,
        weight_matrix: Optional[Sequence[Sequence[float]]] = None,
        temperature: float = 1.0,
        planner: Optional[StructuredPlanner] = None,
        fallback: Optional[CognitivePolicy] = None,
        training_corpus: Optional[Sequence[Dict[str, Any]]] = None,
        learning_rate: float = 0.05,
        l2: float = 0.01,
        seed: int = 13,
    ) -> None:
        self.temperature = max(0.1, float(temperature))
        self.planner = planner or StructuredPlanner()
        self.adaptive_planner = AdaptivePlanner(self.planner)
        self.fallback = fallback or HeuristicCognitivePolicy()
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None
        self._training_metadata: Dict[str, Any] = {}
        self._rng = np.random.default_rng(seed)
        if weight_matrix is not None:
            self.weight_matrix = [list(row) for row in weight_matrix]
            self._weights = np.asarray(self.weight_matrix, dtype=np.float64)
        else:
            corpus = list(training_corpus or self._default_corpus())
            trained = self._train_from_corpus(
                corpus,
                learning_rate=learning_rate,
                l2=l2,
                epochs=self.TRAINING_EPOCHS,
            )
            self._weights = trained
            self.weight_matrix = trained.tolist()

    @staticmethod
    def _summarise_perception(perception: PerceptionSnapshot) -> Dict[str, float]:
        summary: Dict[str, float] = {}
        for name, payload in perception.modalities.items():
            spikes = payload.get("spike_counts") or []
            total = float(sum(spikes))
            if total > 0.0:
                summary[name] = total
        total_sum = sum(summary.values())
        if total_sum <= 0.0:
            return {name: 0.0 for name in perception.modalities}
        return {name: value / total_sum for name, value in summary.items()}

    def _default_corpus(self) -> List[Dict[str, Any]]:
        def perception(
            vision: Optional[Sequence[float]] = None,
            auditory: Optional[Sequence[float]] = None,
            somatosensory: Optional[Sequence[float]] = None,
            proprioception: Optional[Sequence[float]] = None,
        ) -> PerceptionSnapshot:
            modalities: Dict[str, Dict[str, List[float]]] = {}
            if vision is not None:
                modalities["vision"] = {"spike_counts": list(vision)}
            if auditory is not None:
                modalities["auditory"] = {"spike_counts": list(auditory)}
            if somatosensory is not None:
                modalities["somatosensory"] = {"spike_counts": list(somatosensory)}
            if proprioception is not None:
                modalities["proprioception"] = {"spike_counts": list(proprioception)}
            return PerceptionSnapshot(modalities=modalities)

        def emotion(
            primary: EmotionType,
            valence: float,
            arousal: float,
            dominance: float,
            *,
            intensity: float,
            mood: float,
            bias: Dict[str, float],
        ) -> EmotionSnapshot:
            return EmotionSnapshot(
                primary=primary,
                intensity=intensity,
                mood=mood,
                dimensions={"valence": valence, "arousal": arousal, "dominance": dominance},
                intent_bias=bias,
            )

        joyful = emotion(
            EmotionType.HAPPY,
            0.68,
            0.48,
            0.22,
            intensity=0.65,
            mood=0.3,
            bias={"approach": 0.55, "explore": 0.25, "withdraw": 0.1, "observe": 0.1},
        )
        cautious = emotion(
            EmotionType.ANGRY,
            -0.55,
            0.72,
            -0.35,
            intensity=0.7,
            mood=-0.2,
            bias={"withdraw": 0.55, "approach": 0.15, "observe": 0.2, "explore": 0.1},
        )
        curious = emotion(
            EmotionType.HAPPY,
            0.22,
            0.65,
            0.05,
            intensity=0.55,
            mood=0.15,
            bias={"explore": 0.5, "observe": 0.25, "approach": 0.15, "withdraw": 0.1},
        )
        neutral = emotion(
            EmotionType.NEUTRAL,
            0.05,
            0.35,
            0.05,
            intensity=0.3,
            mood=0.05,
            bias={"observe": 0.6, "approach": 0.15, "withdraw": 0.15, "explore": 0.1},
        )

        baseline_personality = PersonalityProfile(
            openness=0.6,
            conscientiousness=0.55,
            extraversion=0.6,
            agreeableness=0.6,
            neuroticism=0.4,
        )

        corpus: List[Dict[str, Any]] = [
            {
                "intention": "approach",
                "perception": perception(
                    vision=[4.0, 2.5, 1.5],
                    auditory=[1.0, 0.4],
                    somatosensory=[0.2, 0.1],
                ),
                "emotion": joyful,
                "personality": PersonalityProfile(
                    openness=0.7,
                    conscientiousness=0.6,
                    extraversion=0.75,
                    agreeableness=0.72,
                    neuroticism=0.25,
                ),
                "curiosity": CuriosityState(
                    drive=0.62, novelty_preference=0.5, fatigue=0.12, last_novelty=0.48
                ),
                "context": {"safety": 0.72, "social": 0.64, "novelty": 0.35},
                "learning": {"cpu": 0.4, "memory": 0.35},
                "history": [{"intention": "observe", "confidence": 0.45}],
                "augment": 3,
            },
            {
                "intention": "withdraw",
                "perception": perception(
                    vision=[0.3, 0.2],
                    auditory=[0.6, 0.55],
                    somatosensory=[0.5, 0.4],
                ),
                "emotion": cautious,
                "personality": PersonalityProfile(
                    openness=0.45,
                    conscientiousness=0.6,
                    extraversion=0.35,
                    agreeableness=0.5,
                    neuroticism=0.7,
                ),
                "curiosity": CuriosityState(
                    drive=0.28, novelty_preference=0.25, fatigue=0.4, last_novelty=0.2
                ),
                "context": {"threat": 0.88, "safety": 0.15, "fatigue": 0.45},
                "learning": {"cpu": 0.55, "memory": 0.6},
                "history": [{"intention": "withdraw", "confidence": 0.55}],
                "augment": 4,
            },
            {
                "intention": "explore",
                "perception": perception(
                    vision=[1.5, 1.1, 0.9],
                    auditory=[1.4, 1.0, 0.6],
                    somatosensory=[0.4, 0.3],
                ),
                "emotion": curious,
                "personality": PersonalityProfile(
                    openness=0.75,
                    conscientiousness=0.5,
                    extraversion=0.65,
                    agreeableness=0.55,
                    neuroticism=0.35,
                ),
                "curiosity": CuriosityState(
                    drive=0.78, novelty_preference=0.82, fatigue=0.18, last_novelty=0.72
                ),
                "context": {"novelty": 0.82, "safety": 0.55, "social": 0.4},
                "learning": {"cpu": 0.45, "memory": 0.35},
                "history": [{"intention": "explore", "confidence": 0.5}],
                "augment": 5,
            },
            {
                "intention": "observe",
                "perception": perception(
                    vision=[0.6, 0.5, 0.4],
                    auditory=[0.2, 0.1],
                    somatosensory=[0.15, 0.1],
                    proprioception=[0.3, 0.2],
                ),
                "emotion": neutral,
                "personality": baseline_personality,
                "curiosity": CuriosityState(
                    drive=0.35, novelty_preference=0.32, fatigue=0.25, last_novelty=0.28
                ),
                "context": {"safety": 0.55, "threat": 0.2, "novelty": 0.25},
                "learning": {"cpu": 0.7, "memory": 0.65},
                "history": [{"intention": "observe", "confidence": 0.6}],
                "augment": 2,
            },
            {
                "intention": "approach",
                "perception": perception(
                    vision=[3.5, 3.1, 2.8],
                    auditory=[0.5, 0.4],
                    somatosensory=[0.4, 0.2],
                ),
                "emotion": joyful,
                "personality": PersonalityProfile(
                    openness=0.68,
                    conscientiousness=0.58,
                    extraversion=0.7,
                    agreeableness=0.66,
                    neuroticism=0.22,
                ),
                "curiosity": CuriosityState(
                    drive=0.58, novelty_preference=0.46, fatigue=0.18, last_novelty=0.42
                ),
                "context": {"safety": 0.68, "social": 0.58, "novelty": 0.4},
                "learning": {"cpu": 0.35, "memory": 0.3},
                "history": [{"intention": "approach", "confidence": 0.55}],
                "augment": 2,
            },
            {
                "intention": "withdraw",
                "perception": perception(
                    vision=[0.4, 0.35],
                    auditory=[0.9, 0.85, 0.8],
                    somatosensory=[0.6, 0.55],
                ),
                "emotion": cautious,
                "personality": PersonalityProfile(
                    openness=0.4,
                    conscientiousness=0.65,
                    extraversion=0.3,
                    agreeableness=0.5,
                    neuroticism=0.75,
                ),
                "curiosity": CuriosityState(
                    drive=0.25, novelty_preference=0.2, fatigue=0.5, last_novelty=0.18
                ),
                "context": {"threat": 0.92, "safety": 0.12, "novelty": 0.15},
                "learning": {"cpu": 0.6, "memory": 0.55},
                "history": [{"intention": "withdraw", "confidence": 0.58}],
                "augment": 3,
            },
            {
                "intention": "explore",
                "perception": perception(
                    vision=[1.2, 1.0, 0.8],
                    auditory=[1.3, 1.1, 0.8],
                    somatosensory=[0.5, 0.4],
                ),
                "emotion": curious,
                "personality": PersonalityProfile(
                    openness=0.78,
                    conscientiousness=0.52,
                    extraversion=0.68,
                    agreeableness=0.6,
                    neuroticism=0.3,
                ),
                "curiosity": CuriosityState(
                    drive=0.82, novelty_preference=0.88, fatigue=0.22, last_novelty=0.75
                ),
                "context": {"novelty": 0.86, "safety": 0.52, "social": 0.35},
                "learning": {"cpu": 0.42, "memory": 0.34},
                "history": [{"intention": "explore", "confidence": 0.52}],
                "augment": 4,
            },
            {
                "intention": "observe",
                "perception": perception(
                    vision=[0.5, 0.45],
                    auditory=[0.3, 0.25],
                    somatosensory=[0.25, 0.2],
                    proprioception=[0.35, 0.28],
                ),
                "emotion": neutral,
                "personality": baseline_personality,
                "curiosity": CuriosityState(
                    drive=0.32, novelty_preference=0.3, fatigue=0.3, last_novelty=0.22
                ),
                "context": {"safety": 0.6, "threat": 0.18, "novelty": 0.2},
                "learning": {"cpu": 0.75, "memory": 0.68},
                "history": [{"intention": "observe", "confidence": 0.58}],
                "augment": 1,
            },
        ]
        return corpus

    def _build_feature_vector(
        self,
        perception: PerceptionSnapshot,
        summary: Dict[str, float],
        emotion: EmotionSnapshot,
        personality: PersonalityProfile,
        curiosity: CuriosityState,
        context: Dict[str, Any],
        learning_prediction: Optional[Dict[str, float]],
        history: Optional[Sequence[Dict[str, Any]]],
    ) -> List[float]:
        valence = emotion.dimensions.get("valence", 0.0)
        arousal = emotion.dimensions.get("arousal", 0.0)
        dominance = emotion.dimensions.get("dominance", 0.0)
        intent_bias = emotion.intent_bias or {}
        context_threat = float(context.get("threat", 0.0))
        context_safety = float(context.get("safety", 0.0))
        context_novelty = float(context.get("novelty", 0.0))
        context_social = float(context.get("social", 0.0))
        context_control = float(context.get("control", 0.0))
        context_fatigue = float(context.get("fatigue", 0.0))
        modalities_count = len(perception.modalities)
        standard_keys = {"vision", "auditory", "somatosensory", "proprioception"}
        other_values = [value for key, value in summary.items() if key not in standard_keys]
        summary_other = sum(other_values) / len(other_values) if other_values else 0.0
        history = history or []
        history_counts = {intent: 0.0 for intent in self.INTENTIONS}
        confidences: List[float] = []
        for item in history:
            if not isinstance(item, dict):
                continue
            intention = str(item.get("intention", "")).lower()
            if intention in history_counts:
                history_counts[intention] += 1.0
            value = item.get("confidence")
            if value is not None:
                try:
                    confidences.append(float(value))
                except (TypeError, ValueError):
                    continue
        total_history = sum(history_counts.values()) or 1.0
        for key in history_counts:
            history_counts[key] /= total_history
        history_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        novelty_delta = curiosity.last_novelty - 0.5
        safety_margin = context_safety - context_threat
        valence_arousal = valence * arousal
        threat_dominance = context_threat * (1.0 - max(0.0, dominance))
        mood_valence = emotion.mood * valence

        return [
            1.0,
            valence,
            arousal,
            dominance,
            emotion.intensity,
            emotion.mood,
            float(intent_bias.get("approach", 0.0)),
            float(intent_bias.get("withdraw", 0.0)),
            float(intent_bias.get("explore", 0.0)),
            float(intent_bias.get("soothe", 0.0)),
            context_threat,
            context_safety,
            context_novelty,
            context_social,
            context_control,
            context_fatigue,
            curiosity.drive,
            curiosity.fatigue,
            curiosity.novelty_preference,
            summary.get("vision", 0.0),
            summary.get("auditory", 0.0),
            summary.get("somatosensory", 0.0),
            summary.get("proprioception", 0.0),
            summary_other,
            min(1.0, modalities_count / 5.0),
            personality.openness,
            personality.conscientiousness,
            personality.extraversion,
            personality.agreeableness,
            personality.neuroticism,
            float(learning_prediction.get("cpu", 0.0)) if learning_prediction else 0.0,
            float(learning_prediction.get("memory", 0.0)) if learning_prediction else 0.0,
            history_counts["approach"],
            history_counts["withdraw"],
            history_counts["explore"],
            history_counts["observe"],
            history_confidence,
            novelty_delta,
            safety_margin,
            valence_arousal,
            threat_dominance,
            mood_valence,
        ]

    def _train_from_corpus(
        self,
        corpus: Sequence[Dict[str, Any]],
        *,
        learning_rate: float,
        l2: float,
        epochs: int,
    ) -> np.ndarray:
        feature_vectors: List[np.ndarray] = []
        labels: List[int] = []
        for sample in corpus:
            perception = sample["perception"]
            summary = self._summarise_perception(perception)
            vector = np.asarray(
                self._build_feature_vector(
                    perception,
                    summary,
                    sample["emotion"],
                    sample["personality"],
                    sample["curiosity"],
                    sample.get("context", {}),
                    sample.get("learning"),
                    sample.get("history"),
                ),
                dtype=np.float64,
            )
            label_index = self.INTENTIONS.index(sample["intention"])
            feature_vectors.append(vector)
            labels.append(label_index)
            augment = int(sample.get("augment", 0))
            for _ in range(max(0, augment)):
                noise = self._rng.normal(0.0, 0.015, size=vector.size)
                noise[0] = 0.0
                feature_vectors.append(vector + noise)
                labels.append(label_index)
        matrix = np.vstack(feature_vectors)
        label_array = np.asarray(labels, dtype=int)
        if matrix.shape[0] < len(self.INTENTIONS):
            raise ValueError("Insufficient training samples for production policy")
        mean = matrix[:, 1:].mean(axis=0)
        std = matrix[:, 1:].std(axis=0)
        std[std < 1e-6] = 1.0
        self._feature_mean = mean
        self._feature_std = std
        norm_matrix = matrix.copy()
        norm_matrix[:, 1:] = (norm_matrix[:, 1:] - mean) / std
        n_samples, n_features = norm_matrix.shape
        n_classes = len(self.INTENTIONS)
        weights = self._rng.normal(0.0, 0.05, size=(n_classes, n_features))
        one_hot = np.zeros((n_samples, n_classes), dtype=np.float64)
        one_hot[np.arange(n_samples), label_array] = 1.0
        for _ in range(max(1, epochs)):
            logits = norm_matrix @ weights.T
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
            grad = (probs - one_hot).T @ norm_matrix / n_samples
            weights -= learning_rate * (grad + l2 * weights)
        self._training_metadata = {
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "l2": float(l2),
            "samples": int(n_samples),
        }
        return weights

    def _prepare_features(self, feature_vector: Sequence[float]) -> np.ndarray:
        arr = np.asarray(feature_vector, dtype=np.float64)
        if self._feature_mean is not None and self._feature_std is not None:
            if arr.size - 1 != self._feature_mean.size:
                raise ValueError("feature normalisation mismatch")
            arr[1:] = (arr[1:] - self._feature_mean) / (self._feature_std + 1e-6)
        return arr

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        logits = logits.astype(np.float64)
        logits -= logits.max()
        exp = np.exp(logits / self.temperature)
        denom = exp.sum()
        if denom <= 0.0:
            return np.full(logits.shape, 1.0 / logits.size)
        return exp / denom

    def select_intention(
        self,
        perception: PerceptionSnapshot,
        summary: Dict[str, float],
        emotion: EmotionSnapshot,
        personality: PersonalityProfile,
        curiosity: CuriosityState,
        context: Dict[str, Any],
        learning_prediction: Optional[Dict[str, float]] = None,
        history: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> CognitiveDecision:
        try:
            feature_vector = self._prepare_features(
                self._build_feature_vector(
                    perception,
                    summary,
                    emotion,
                    personality,
                    curiosity,
                    context,
                    learning_prediction,
                    history,
                )
            )
            if feature_vector.size != self._weights.shape[1]:
                raise ValueError("feature dimension mismatch")
            logits = self._weights @ feature_vector
            probabilities = self._softmax(logits)
            index = int(np.argmax(probabilities))
            intention = self.INTENTIONS[index]
            confidence = float(probabilities[index])
            focus = max(summary, key=summary.get) if summary else None
            plan_steps = self.adaptive_planner.generate(
                intention,
                focus,
                context,
                perception,
                summary,
                emotion,
                curiosity,
                history,
                learning_prediction,
            )
            weights = {
                intent: float(probabilities[i]) for i, intent in enumerate(self.INTENTIONS)
            }
            tags = ["policy-production", intention]
            if confidence >= 0.65:
                tags.append("high-confidence")
            if curiosity.drive > 0.6 or context.get("novelty", 0.0) > 0.6:
                tags.append("novelty-driven")
            if focus:
                tags.append(f"focus-{focus}")
            thought_trace = [
                f"features={feature_vector.size}",
                f"intention={intention}",
                f"confidence={confidence:.2f}",
                f"valence={emotion.dimensions.get('valence', 0.0):.2f}",
                f"novelty={context.get('novelty', curiosity.last_novelty):.2f}",
            ]
            metadata = {
                "policy": "production",
                "policy_version": "2.0",
                "planner": getattr(
                    self.adaptive_planner.base,
                    "name",
                    self.adaptive_planner.base.__class__.__name__.lower(),
                ),
                "temperature": self.temperature,
                "logits": [float(value) for value in logits],
                "probabilities": weights,
                "training": dict(self._training_metadata),
            }
            summary_text = ", ".join(f"{k}:{v:.2f}" for k, v in summary.items()) or "no-salient-modalities"
            return CognitiveDecision(
                intention=intention,
                confidence=confidence,
                plan=list(plan_steps),
                weights=weights,
                tags=tags,
                focus=focus,
                summary=summary_text,
                thought_trace=thought_trace,
                perception_summary=dict(summary),
                metadata=metadata,
            )
        except Exception as exc:  # pragma: no cover - defensive safeguard
            logger.debug("Production policy failed: %s", exc)
            decision = self.fallback.select_intention(
                perception,
                summary,
                emotion,
                personality,
                curiosity,
                context,
                learning_prediction,
                history=history,
            )
            decision.metadata.setdefault("policy", "production-fallback")
            decision.metadata["policy_error"] = str(exc)
            return decision


class CognitiveModule:
    """Stateful cognitive reasoning module with adaptive planning fallbacks."""

    def __init__(
        self,
        memory_window: int = 8,
        policy: Optional[CognitivePolicy] = None,
    ) -> None:
        self.memory_window = memory_window
        self.episodic_memory: deque[dict[str, Any]] = deque(maxlen=memory_window)
        self.policy: CognitivePolicy = policy or ProductionCognitivePolicy()
        self._confidence_history: deque[float] = deque(maxlen=max(4, memory_window))
        if isinstance(getattr(self.policy, "adaptive_planner", None), AdaptivePlanner):
            self._fallback_planner = self.policy.adaptive_planner
        else:
            self._fallback_planner = AdaptivePlanner()

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
                history=list(self.episodic_memory),
            )
        except Exception as exc:  # pragma: no cover - defensive policy fallback
            logger.warning("Cognitive policy failed; using fallback plan. Error: %s", exc)
            intention = context.get("fallback_intention", "observe")
            focus = max(summary, key=summary.get) if summary else None
            fallback_plan = self._fallback_planner.generate(
                intention,
                focus,
                context,
                perception,
                summary,
                emotion,
                curiosity,
                history=list(self.episodic_memory),
                learning_prediction=learning_prediction,
            )
            policy_decision = CognitiveDecision(
                intention=intention,
                confidence=0.25,
                plan=fallback_plan,
                weights={intention: 1.0},
                tags=[intention, "policy-fallback"],
                focus=focus,
                summary=", ".join(f"{k}:{v:.2f}" for k, v in summary.items())
                or "no-salient-modalities",
                thought_trace=["policy=fallback", f"error={exc}"],
                perception_summary=dict(summary),
                metadata={"policy": "fallback", "error": str(exc)},
            )

        if not policy_decision.plan:
            policy_decision.plan = self._fallback_planner.generate(
                policy_decision.intention,
                policy_decision.focus,
                context,
                perception,
                summary,
                emotion,
                curiosity,
                history=list(self.episodic_memory),
                learning_prediction=learning_prediction,
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
