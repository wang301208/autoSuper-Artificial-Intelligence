"""Data-driven limbic system components for emotion processing.

This module provides deployable substitutes for brain regions typically
associated with emotional processing.  The goal is to keep the API
practical for large-scale agents while implementing multi-dimensional
valence–arousal–dominance modelling and homeostatic regulation hooks that
are exercised throughout the cognitive stack.

The :class:`LimbicSystem` orchestrates three sub-modules:

``EmotionProcessor``
    Derives a primary emotion from a textual stimulus.
``MemoryConsolidator``
    Stores past stimulus/response pairs for later inspection.
``HomeostasisController``
    Keeps the emotional intensity within a bounded range while adjusting mood.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Tuple

from schemas.emotion import EmotionalState, EmotionType

from .state import BrainRuntimeConfig, PersonalityProfile


class EmotionProcessor:
    """Data-driven emotion classifier using lightweight VAD regression."""

    DEFAULT_VAD_LEXICON: Dict[str, Tuple[float, float, float]] = {
        "good": (0.72, 0.46, 0.54),
        "great": (0.85, 0.62, 0.60),
        "happy": (0.89, 0.72, 0.55),
        "joy": (0.95, 0.78, 0.55),
        "love": (0.93, 0.74, 0.52),
        "calm": (0.60, 0.20, 0.45),
        "relaxed": (0.70, 0.25, 0.55),
        "relief": (0.70, 0.32, 0.40),
        "secure": (0.68, 0.28, 0.58),
        "proud": (0.82, 0.50, 0.70),
        "win": (0.86, 0.65, 0.70),
        "success": (0.80, 0.58, 0.60),
        "bad": (-0.60, 0.50, -0.50),
        "sad": (-0.75, 0.45, -0.40),
        "angry": (-0.82, 0.78, 0.45),
        "terrible": (-0.85, 0.75, -0.60),
        "fail": (-0.74, 0.60, -0.55),
        "loss": (-0.70, 0.52, -0.50),
        "fear": (-0.60, 0.85, -0.60),
        "panic": (-0.70, 0.90, -0.70),
        "worry": (-0.55, 0.60, -0.55),
        "furious": (-0.80, 0.85, 0.30),
        "threat": (-0.70, 0.82, -0.60),
        "danger": (-0.70, 0.86, -0.55),
        "disgust": (-0.80, 0.68, -0.65),
        "grief": (-0.72, 0.50, -0.60),
        "excited": (0.88, 0.85, 0.60),
        "energized": (0.85, 0.90, 0.55),
        "surprise": (0.20, 0.75, -0.10),
        "bored": (-0.30, 0.20, -0.30),
        "resent": (-0.70, 0.60, -0.40),
        "hope": (0.78, 0.48, 0.55),
        "peace": (0.76, 0.25, 0.58),
        "support": (0.74, 0.32, 0.52),
    }
    INTENSIFIERS = {"very": 0.35, "extremely": 0.50, "incredibly": 0.45, "super": 0.20, "really": 0.20, "so": 0.15}
    NEGATIONS = {"not", "never", "no", "hardly", "barely"}
    ACTIVATION_TERMS = {
        "urgent",
        "surprise",
        "alert",
        "shock",
        "excited",
        "furious",
        "panic",
        "energized",
        "thrill",
        "alarm",
        "intense",
        "pressure",
    }

    def __init__(self, lexicon: Dict[str, Tuple[float, float, float]] | None = None) -> None:
        self.vad_lexicon = dict(lexicon or self.DEFAULT_VAD_LEXICON)
        self.baseline = {"valence": 0.0, "arousal": 0.35, "dominance": 0.05}
        self.positive_terms = {term for term, (valence, _, _) in self.vad_lexicon.items() if valence > 0.35}
        self.negative_terms = {term for term, (valence, _, _) in self.vad_lexicon.items() if valence < -0.35}

    def _tokenize(self, stimulus: str) -> List[str]:
        return re.findall(r"[\w']+", stimulus.lower())

    def _aggregate_vad(self, tokens: List[str]) -> Dict[str, float]:
        totals = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
        total_weight = 0.0
        negated = False
        pending_intensity = 0.0
        for token in tokens:
            if token in self.INTENSIFIERS:
                pending_intensity = self.INTENSIFIERS[token]
                continue
            if token in self.NEGATIONS:
                negated = not negated
                continue
            if token not in self.vad_lexicon:
                pending_intensity = 0.0
                continue
            weight = 1.0 + pending_intensity
            pending_intensity = 0.0
            valence, arousal, dominance = self.vad_lexicon[token]
            if negated:
                valence = -valence * 0.85
                dominance = -dominance * 0.65
                arousal = max(0.0, arousal * 0.75)
                negated = False
            totals["valence"] += valence * weight
            totals["arousal"] += arousal * weight
            totals["dominance"] += dominance * weight
            total_weight += weight
        if total_weight:
            return {k: v / total_weight for k, v in totals.items()}
        return dict(self.baseline)

    def _textual_features(self, tokens: List[str], stimulus: str) -> Dict[str, float]:
        positive_hits = sum(1 for token in tokens if token in self.positive_terms)
        negative_hits = sum(1 for token in tokens if token in self.negative_terms)
        sentiment_total = positive_hits + negative_hits
        lexical_sentiment = 0.0
        if sentiment_total:
            lexical_sentiment = (positive_hits - negative_hits) / sentiment_total
        activation_hits = sum(1 for token in tokens if token in self.ACTIVATION_TERMS)
        exclaim_count = stimulus.count("!")
        question_count = stimulus.count("?")
        emphasis_count = sum(
            1 for word in re.findall(r"\b\w+\b", stimulus) if word.isupper() and len(word) > 2
        )
        negation_hits = sum(1 for token in tokens if token in self.NEGATIONS)
        lexical_intensity = min(
            1.0, activation_hits * 0.25 + exclaim_count * 0.08 + emphasis_count * 0.12
        )
        return {
            "sentiment": lexical_sentiment,
            "intensity": lexical_intensity,
            "activation": min(1.0, activation_hits * 0.2),
            "negation_density": negation_hits / max(1, len(tokens)),
            "emphasis": min(1.0, emphasis_count * 0.25),
            "questions": min(1.0, question_count * 0.25),
            "coverage": sentiment_total / max(1, len(tokens)),
        }

    def _normalize_context(self, context: Dict[str, float] | None) -> Dict[str, float]:
        if not context:
            return {}
        normalized: Dict[str, float] = {}
        for key, value in context.items():
            try:
                normalized[key] = max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                continue
        return normalized

    def evaluate(
        self, stimulus: str, context: Dict[str, float] | None = None
    ) -> Tuple[EmotionType, Dict[str, float], Dict[str, float]]:
        tokens = self._tokenize(stimulus)
        lexical_vad = self._aggregate_vad(tokens)
        features = self._textual_features(tokens, stimulus)
        context_weights = self._normalize_context(context)

        novelty = context_weights.get("novelty", 0.0)
        safety = context_weights.get("safety", 0.0)
        threat = context_weights.get("threat", 0.0)
        social = context_weights.get("social", 0.0)
        control = context_weights.get("control", 0.0)
        fatigue = context_weights.get("fatigue", 0.0)

        lexical_sentiment = features["sentiment"]
        sentiment_weight = 0.35 + features["coverage"] * 0.25
        negation_density = features["negation_density"]
        lexical_intensity = features["intensity"]

        valence_raw = (
            self.baseline["valence"]
            + lexical_vad["valence"] * 0.75
            + lexical_sentiment * sentiment_weight
            - negation_density * 0.30
            + safety * 0.45
            - threat * 0.60
            + social * 0.22
            + control * 0.20
            - fatigue * 0.22
        )
        valence_raw += lexical_intensity * 0.05
        valence = math.tanh(valence_raw)

        arousal_raw = (
            self.baseline["arousal"]
            + lexical_vad["arousal"] * 0.65
            + lexical_intensity * 0.50
            + novelty * 0.45
            + threat * 0.50
            - safety * 0.15
            + features["activation"] * 0.25
            + features["emphasis"] * 0.20
        )
        arousal_raw -= fatigue * 0.30
        arousal = max(0.0, min(1.0, arousal_raw))

        dominance_raw = (
            self.baseline["dominance"]
            + lexical_vad["dominance"] * 0.60
            + lexical_sentiment * 0.20
            + control * 0.35
            + safety * 0.25
            - threat * 0.45
            + social * 0.18
            - features["questions"] * 0.15
            - negation_density * 0.25
        )
        dominance_raw += (valence - self.baseline["valence"]) * 0.25
        dominance = math.tanh(dominance_raw)

        dimensions = {
            "valence": max(-1.0, min(1.0, valence)),
            "arousal": arousal,
            "dominance": max(-1.0, min(1.0, dominance)),
        }

        emotion = EmotionType.NEUTRAL
        if valence >= 0.25:
            if arousal >= 0.30 or dominance >= -0.05:
                emotion = EmotionType.HAPPY
        elif valence <= -0.25:
            if threat >= 0.45 or dominance >= 0.10 or arousal >= 0.65:
                emotion = EmotionType.ANGRY
            else:
                emotion = EmotionType.SAD
        else:
            if threat >= 0.65 or (arousal >= 0.70 and dominance > 0.0):
                emotion = EmotionType.ANGRY
            elif valence <= -0.10 or (
                lexical_sentiment < -0.20 and features["coverage"] > 0.20
            ):
                emotion = EmotionType.SAD

        return emotion, dimensions, context_weights


class MemoryConsolidator:
    """Store stimulus and emotion pairs for rudimentary memory."""

    def __init__(self, max_items: int = 64) -> None:
        self.max_items = max_items
        self.memory: List[Tuple[str, EmotionType, Dict[str, float]]] = []

    def consolidate(self, stimulus: str, emotion: EmotionType, dimensions: Dict[str, float]) -> None:
        self.memory.append((stimulus, emotion, dict(dimensions)))
        if len(self.memory) > self.max_items:
            self.memory.pop(0)

    def recent(self, limit: int = 5) -> List[Tuple[str, EmotionType, Dict[str, float]]]:
        return self.memory[-limit:]


class HomeostasisController:
    """Regulate the intensity of an :class:`EmotionalState` and track mood."""

    def __init__(self, decay_rate: float = 0.15, set_point: Dict[str, float] | None = None) -> None:
        self.mood: float = 0.0  # range [-1, 1]
        self.decay_rate = max(0.0, min(1.0, decay_rate))
        self.set_point = set_point or {"valence": 0.05, "arousal": 0.35, "dominance": 0.05}
        self.mood_inertia = 1.0 - self.decay_rate * 0.6
        self.last_dimensions: Dict[str, float] = dict(self.set_point)

    def regulate(
        self,
        state: EmotionalState,
        personality: PersonalityProfile,
        context: Dict[str, float],
        enable_decay: bool,
        full_dimensions: Dict[str, float] | None = None,
    ) -> EmotionalState:
        if enable_decay:
            self.mood *= self.mood_inertia
        else:
            self.mood *= 1.0 - self.decay_rate * 0.2

        dims_source = full_dimensions or state.dimensions
        valence = dims_source.get("valence", self.set_point["valence"])
        arousal = dims_source.get("arousal", state.intensity if "arousal" not in dims_source else dims_source["arousal"])
        dominance = dims_source.get("dominance", self.set_point["dominance"])
        self.last_dimensions = {
            "valence": valence,
            "arousal": arousal,
            "dominance": dominance,
        }

        safety = context.get("safety", 0.0)
        threat = context.get("threat", 0.0)
        social = context.get("social", 0.0)
        novelty = context.get("novelty", 0.0)
        fatigue = context.get("fatigue", 0.0)

        valence_error = valence - self.set_point["valence"]
        arousal_error = arousal - self.set_point["arousal"]
        dominance_error = dominance - self.set_point["dominance"]

        mood_delta = valence_error * (0.48 + personality.extraversion * 0.30)
        mood_delta -= max(0.0, -valence_error) * (0.38 + personality.neuroticism * 0.35)
        mood_delta += dominance_error * 0.25
        mood_delta -= abs(arousal_error) * 0.10
        mood_delta += safety * 0.12 + social * 0.08
        mood_delta -= threat * (0.28 + personality.neuroticism * 0.20)
        self.mood = max(-1.0, min(1.0, self.mood + mood_delta))

        affective_drive = (
            abs(valence_error) * 0.45
            + max(0.0, arousal_error) * 0.35
            + abs(dominance_error) * 0.25
        )
        context_drive = (
            threat * (0.40 + personality.neuroticism * 0.30)
            + novelty * (0.30 + personality.openness * 0.20)
            + safety * (0.10 + personality.agreeableness * 0.10)
        )
        mood_drive = abs(self.mood) * 0.30
        dominance_drive = max(0.0, dominance_error) * 0.20

        intensity_target = 0.25 + affective_drive + context_drive + mood_drive + dominance_drive
        if valence < self.set_point["valence"]:
            intensity_target += personality.neuroticism * 0.20
        else:
            intensity_target += personality.extraversion * 0.15
        intensity_target -= fatigue * 0.20
        intensity_target = max(0.0, min(1.0, intensity_target))

        smoothing = 0.40 + self.decay_rate * 0.30 if enable_decay else 0.30
        smoothing = min(0.95, max(0.10, smoothing))
        state.intensity = max(0.0, min(1.0, state.intensity * (1 - smoothing) + intensity_target * smoothing))
        state.decay = self.decay_rate if enable_decay else 0.0
        return state


class LimbicSystem:
    """High level facade coordinating limbic sub-modules."""

    def __init__(self, personality: PersonalityProfile | None = None) -> None:
        self.emotion_processor = EmotionProcessor()
        self.memory_consolidator = MemoryConsolidator()
        self.homeostasis_controller = HomeostasisController()
        self.personality = personality or PersonalityProfile()
        self.personality.clamp()

    def react(self, stimulus: str, context: Dict[str, float] | None = None, config: BrainRuntimeConfig | None = None) -> EmotionalState:
        """Process ``stimulus`` and return the resulting emotional state."""

        config = config or BrainRuntimeConfig()
        emotion, full_dimensions, context_weights = self.emotion_processor.evaluate(stimulus, context)
        full_dimensions = dict(full_dimensions)
        if config.enable_multi_dim_emotion:
            dimensions = dict(full_dimensions)
        else:
            dimensions = {"valence": full_dimensions.get("valence", 0.0)}

        valence = full_dimensions.get("valence", 0.0)
        arousal = full_dimensions.get("arousal", 0.35)
        dominance = full_dimensions.get("dominance", 0.05)
        base_intensity = 0.28 + abs(valence) * 0.32 + arousal * 0.30 + max(0.0, -dominance) * 0.08
        if emotion != EmotionType.NEUTRAL:
            base_intensity += 0.10
        base_intensity = max(0.0, min(1.0, base_intensity))
        if config.enable_personality_modulation:
            if valence >= 0:
                base_intensity += self.personality.extraversion * 0.10
            else:
                base_intensity += self.personality.neuroticism * 0.12
            base_intensity += (self.personality.openness - 0.5) * 0.06 * (arousal - 0.35)
            base_intensity += (self.personality.agreeableness - 0.5) * 0.04 * context_weights.get("social", 0.0)
            base_intensity = max(0.0, min(1.0, base_intensity))

        state = EmotionalState(
            emotion=emotion,
            intensity=base_intensity,
            dimensions=dimensions,
            context_weights=context_weights,
        )
        state = self.homeostasis_controller.regulate(
            state,
            self.personality if config.enable_personality_modulation else PersonalityProfile(),
            context_weights,
            enable_decay=config.enable_emotion_decay,
            full_dimensions=full_dimensions,
        )
        state.intent_bias = self._intent_bias(
            state, context_weights, config, full_dimensions=full_dimensions
        )
        self.memory_consolidator.consolidate(stimulus, emotion, full_dimensions)
        return state

    def _intent_bias(
        self,
        state: EmotionalState,
        context: Dict[str, float],
        config: BrainRuntimeConfig,
        full_dimensions: Dict[str, float] | None = None,
    ) -> Dict[str, float]:
        dims = full_dimensions or state.dimensions
        valence = dims.get("valence", state.dimensions.get("valence", 0.0))
        arousal = dims.get("arousal", state.dimensions.get("arousal", 0.0))
        dominance = dims.get("dominance", 0.0)
        novelty = context.get("novelty", 0.0)
        threat = context.get("threat", 0.0)
        safety = context.get("safety", 0.0)
        fatigue = context.get("fatigue", 0.0)

        approach_drive = max(0.0, 0.45 + 0.45 * valence + 0.20 * dominance + max(0.0, arousal - 0.4) * 0.20)
        withdraw_drive = max(
            0.0,
            0.45 - 0.35 * valence - 0.15 * dominance + threat * 0.50 + max(0.0, -dominance) * 0.20,
        )
        explore_drive = max(0.0, 0.30 + 0.55 * arousal + 0.30 * novelty - fatigue * 0.25)
        soothe_drive = max(0.0, 0.25 + 0.30 * (1 - arousal) + max(0.0, -valence) * 0.20 + safety * 0.20)

        bias = {
            "approach": approach_drive,
            "withdraw": withdraw_drive,
            "explore": explore_drive,
            "soothe": soothe_drive,
        }
        if config.enable_personality_modulation:
            bias["approach"] *= 0.55 + self.personality.extraversion * 0.45
            bias["withdraw"] *= 0.55 + self.personality.neuroticism * 0.45
            bias["explore"] *= 0.50 + self.personality.openness * 0.50
            bias["soothe"] *= 0.55 + (
                0.5 * self.personality.agreeableness + 0.5 * self.personality.conscientiousness
            )
        total = sum(bias.values()) or 1.0
        return {k: min(1.0, max(0.0, v / total)) for k, v in bias.items()}

    @property
    def mood(self) -> float:
        return self.homeostasis_controller.mood

    def update_personality(self, profile: PersonalityProfile) -> None:
        profile.clamp()
        self.personality = profile


__all__ = [
    "EmotionProcessor",
    "MemoryConsolidator",
    "HomeostasisController",
    "LimbicSystem",
]

