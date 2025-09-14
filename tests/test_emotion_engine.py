import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from emotion_engine import EmotionEngine
from schemas.emotion import EmotionalState, EmotionType


def test_emotion_engine_process_emotion():
    engine = EmotionEngine()
    state = engine.process_emotion("I feel good today")
    assert isinstance(state, EmotionalState)
    assert isinstance(state.emotion, EmotionType)
    assert 0.0 <= state.intensity <= 1.0

