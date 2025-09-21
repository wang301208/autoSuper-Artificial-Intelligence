"""Simplified limbic system components for emotion processing.

This module contains lightweight stand-ins for brain regions typically
associated with emotional processing.  The goal of these classes is not to
model neuroscience accurately but to provide a small, easily testable API
that other parts of the project can interact with.

The :class:`LimbicSystem` orchestrates three sub-modules:

``EmotionProcessor``
    Derives a primary emotion from a textual stimulus.
``MemoryConsolidator``
    Stores past stimulus/response pairs for later inspection.
``HomeostasisController``
    Keeps the emotional intensity within a bounded range while adjusting mood.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from schemas.emotion import EmotionalState, EmotionType

from .state import BrainRuntimeConfig, PersonalityProfile


class EmotionProcessor:
    """Very small heuristic based emotion classifier with context support."""

    POSITIVE_KEYWORDS = {"good", "happy", "love", "great", "win", "success"}
    NEGATIVE_KEYWORDS = {"bad", "sad", "angry", "terrible", "fail", "loss"}
    ACTIVATION_KEYWORDS = {"urgent", "surprise", "alert", "shock"}

    def evaluate(self, stimulus: str, context: Dict[str, float] | None = None) -> Tuple[EmotionType, Dict[str, float], Dict[str, float]]:
        lowered = stimulus.lower()
        valence = 0.0
        arousal = 0.2
        dominance = 0.0

        if any(word in lowered for word in self.POSITIVE_KEYWORDS):
            valence += 0.6
            arousal += 0.2
        if any(word in lowered for word in self.NEGATIVE_KEYWORDS):
            valence -= 0.7
            arousal += 0.3
        if any(word in lowered for word in self.ACTIVATION_KEYWORDS):
            arousal += 0.25

        context = context or {}
        novelty = max(0.0, min(1.0, float(context.get("novelty", 0.0))))
        safety = max(0.0, min(1.0, float(context.get("safety", 0.0))))
        threat = max(0.0, min(1.0, float(context.get("threat", 0.0))))
        social = max(0.0, min(1.0, float(context.get("social", 0.0))))

        valence += safety * 0.4 - threat * 0.6 + social * 0.2
        arousal += novelty * 0.3 + threat * 0.5
        dominance += safety * 0.2 - threat * 0.3 + social * 0.1

        emotion = EmotionType.NEUTRAL
        if valence > 0.35:
            emotion = EmotionType.HAPPY
        elif valence < -0.35:
            emotion = EmotionType.SAD
        elif threat > 0.4:
            emotion = EmotionType.ANGRY

        dimensions = {
            "valence": max(-1.0, min(1.0, valence)),
            "arousal": max(0.0, min(1.0, arousal)),
            "dominance": max(-1.0, min(1.0, dominance)),
        }
        context_weights = {k: max(0.0, min(1.0, float(v))) for k, v in context.items()}
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

    def __init__(self, decay_rate: float = 0.15) -> None:
        self.mood: float = 0.0  # range [-1, 1]
        self.decay_rate = max(0.0, min(1.0, decay_rate))
        self.last_dimensions: Dict[str, float] = {"valence": 0.0, "arousal": 0.2, "dominance": 0.0}

    def regulate(self, state: EmotionalState, personality: PersonalityProfile, context: Dict[str, float], enable_decay: bool) -> EmotionalState:
        if enable_decay:
            self.mood *= 1 - self.decay_rate
        valence = state.dimensions.get("valence", 0.0)
        arousal = state.dimensions.get("arousal", state.intensity)
        safety = context.get("safety", 0.0)
        threat = context.get("threat", 0.0)

        mood_delta = valence * (0.35 + personality.extraversion * 0.3)
        mood_delta -= max(0.0, -valence) * (0.25 + personality.neuroticism * 0.4)
        mood_delta += safety * 0.1 - threat * 0.2
        self.mood = max(-1.0, min(1.0, self.mood + mood_delta))

        intensity_bias = 0.0
        if valence >= 0:
            intensity_bias += 0.2 * personality.extraversion
        else:
            intensity_bias += 0.25 * personality.neuroticism
        intensity_bias += 0.1 * personality.agreeableness * context.get("social", 0.0)

        context_arousal = arousal + threat * 0.4
        if enable_decay:
            state.intensity = state.intensity * (1 - self.decay_rate) + abs(self.mood) * 0.2 + context_arousal * 0.2 + intensity_bias
            state.decay = self.decay_rate
        else:
            state.intensity = state.intensity + abs(self.mood) * 0.1 + context_arousal * 0.1 + intensity_bias
            state.decay = 0.0
        state.intensity = max(0.0, min(1.0, state.intensity))
        self.last_dimensions = dict(state.dimensions)
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
        emotion, dimensions, context_weights = self.emotion_processor.evaluate(stimulus, context)
        if not config.enable_multi_dim_emotion:
            dimensions = {"valence": dimensions.get("valence", 0.0)}
        base_intensity = 0.3 if emotion == EmotionType.NEUTRAL else 0.7
        if config.enable_personality_modulation:
            if emotion == EmotionType.HAPPY:
                base_intensity *= 0.6 + self.personality.extraversion * 0.5
            elif emotion == EmotionType.SAD:
                base_intensity *= 0.6 + self.personality.neuroticism * 0.5
            elif emotion == EmotionType.ANGRY:
                base_intensity *= 0.6 + self.personality.neuroticism * 0.4
            else:
                base_intensity *= 0.5 + (1 - self.personality.neuroticism) * 0.3
        state = EmotionalState(
            emotion=emotion,
            intensity=max(0.0, min(1.0, base_intensity)),
            dimensions=dimensions,
            context_weights=context_weights,
        )
        state = self.homeostasis_controller.regulate(
            state,
            self.personality if config.enable_personality_modulation else PersonalityProfile(),
            context_weights,
            enable_decay=config.enable_emotion_decay,
        )
        state.intent_bias = self._intent_bias(state, context_weights, config)
        self.memory_consolidator.consolidate(stimulus, emotion, dimensions)
        return state

    def _intent_bias(self, state: EmotionalState, context: Dict[str, float], config: BrainRuntimeConfig) -> Dict[str, float]:
        valence = state.dimensions.get("valence", 0.0)
        arousal = state.dimensions.get("arousal", 0.0)
        bias = {
            "approach": max(0.0, min(1.0, (valence + 1) / 2)),
            "withdraw": max(0.0, min(1.0, (-valence + 1) / 2)),
            "explore": max(0.0, min(1.0, arousal * 0.7 + context.get("novelty", 0.0) * 0.6)),
        }
        if config.enable_personality_modulation:
            bias["approach"] *= 0.6 + self.personality.extraversion * 0.4
            bias["withdraw"] *= 0.6 + self.personality.neuroticism * 0.4
            bias["explore"] *= 0.5 + self.personality.openness * 0.5
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

