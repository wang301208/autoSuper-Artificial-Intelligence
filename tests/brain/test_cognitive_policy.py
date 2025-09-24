import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.state import (
    CuriosityState,
    EmotionSnapshot,
    PerceptionSnapshot,
    PersonalityProfile,
)
from modules.brain.whole_brain import CognitiveModule, ProductionCognitivePolicy
from schemas.emotion import EmotionType


def _perception(modalities=None):
    modalities = modalities or {
        "vision": {"spike_counts": [3.0, 2.0, 1.0]},
        "auditory": {"spike_counts": [1.0, 0.5]},
        "somatosensory": {"spike_counts": [0.2, 0.1]},
    }
    return PerceptionSnapshot(modalities=modalities)


def _emotion(primary: EmotionType, *, valence: float, arousal: float, dominance: float, intent_bias):
    return EmotionSnapshot(
        primary=primary,
        intensity=0.6,
        mood=0.2,
        dimensions={"valence": valence, "arousal": arousal, "dominance": dominance},
        context={},
        decay=0.1,
        intent_bias=intent_bias,
    )


def test_production_policy_prefers_approach_under_positive_context():
    policy = ProductionCognitivePolicy()
    perception = _perception()
    summary = {"vision": 0.5, "auditory": 0.3, "somatosensory": 0.2}
    emotion = _emotion(
        EmotionType.HAPPY,
        valence=0.65,
        arousal=0.45,
        dominance=0.25,
        intent_bias={"approach": 0.5, "withdraw": 0.15, "explore": 0.25, "observe": 0.1},
    )
    personality = PersonalityProfile(
        openness=0.7,
        conscientiousness=0.6,
        extraversion=0.75,
        agreeableness=0.7,
        neuroticism=0.2,
    )
    curiosity = CuriosityState(drive=0.65, novelty_preference=0.7, fatigue=0.1, last_novelty=0.55)
    decision = policy.select_intention(
        perception,
        summary,
        emotion,
        personality,
        curiosity,
        {"safety": 0.7, "social": 0.6},
        history=[{"intention": "approach", "confidence": 0.65}],
    )

    assert decision.intention == "approach"
    assert decision.confidence > 0.35
    assert decision.metadata["policy"] == "production"
    assert len(decision.plan) >= 3
    assert decision.weights["approach"] > decision.weights["withdraw"]


def test_production_policy_prioritises_withdraw_with_high_threat():
    policy = ProductionCognitivePolicy()
    perception = _perception()
    summary = {"vision": 0.4, "auditory": 0.4, "somatosensory": 0.2}
    emotion = _emotion(
        EmotionType.ANGRY,
        valence=-0.55,
        arousal=0.65,
        dominance=-0.25,
        intent_bias={"approach": 0.15, "withdraw": 0.5, "explore": 0.2, "observe": 0.15},
    )
    personality = PersonalityProfile(
        openness=0.4,
        conscientiousness=0.55,
        extraversion=0.35,
        agreeableness=0.45,
        neuroticism=0.7,
    )
    curiosity = CuriosityState(drive=0.35, novelty_preference=0.4, fatigue=0.25, last_novelty=0.4)
    decision = policy.select_intention(
        perception,
        summary,
        emotion,
        personality,
        curiosity,
        {"threat": 0.9, "safety": 0.1},
        history=[{"intention": "withdraw", "confidence": 0.6}],
    )

    assert decision.intention == "withdraw"
    assert decision.weights["withdraw"] > decision.weights["approach"]
    assert decision.metadata["policy"] == "production"
    assert decision.plan[0] in {"elevate_alert_state", "assess_risk_vectors"}


def test_cognitive_module_uses_production_policy_default():
    module = CognitiveModule()
    assert isinstance(module.policy, ProductionCognitivePolicy)

    perception = _perception()
    emotion = _emotion(
        EmotionType.HAPPY,
        valence=0.45,
        arousal=0.35,
        dominance=0.1,
        intent_bias={"approach": 0.4, "withdraw": 0.2, "explore": 0.25, "observe": 0.15},
    )
    personality = PersonalityProfile()
    curiosity = CuriosityState()
    decision = module.decide(
        perception,
        emotion,
        personality,
        curiosity,
        context={"safety": 0.5, "novelty": 0.4},
    )

    assert decision["policy_metadata"]["policy"] == "production"
    assert decision["plan"]
    assert 0.0 <= decision["confidence"] <= 1.0
