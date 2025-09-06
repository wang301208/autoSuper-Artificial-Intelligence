import os
import sys

sys.path.insert(0, os.path.abspath(os.getcwd()))

from modules.common.emotion import (
    EmotionAnalyzer,
    EmotionProfile,
    EmotionState,
    adjust_response_style,
)


def test_keyword_model_classifies_voice():
    analyzer = EmotionAnalyzer()
    assert analyzer.analyze_voice(b"I love this") == "positive"


def test_emotion_analyzer_classifies_basic_sentiment():
    analyzer = EmotionAnalyzer()
    assert analyzer.analyze_text("I love this!") == "positive"
    assert analyzer.analyze_text("This is terrible") == "negative"
    assert analyzer.analyze_text("It is a table") == "neutral"


def test_emotion_profile_influences_classification_and_style():
    profile = EmotionProfile(positive_threshold=2, positive_suffix=":)", negative_prefix="Apologies:")
    analyzer = EmotionAnalyzer(profile=profile)
    state = EmotionState()

    state.update("good", analyzer)
    assert state.label == "neutral"  # not enough positive keywords

    state.update("good great", analyzer)
    assert state.label == "positive"

    response = adjust_response_style("Thanks", state, profile)
    assert response.endswith(":)")


def test_adjust_response_style_handles_multimodal_signals():
    analyzer = EmotionAnalyzer()
    state = EmotionState()
    state.update("good", analyzer)

    # Voice indicates negativity twice; majority vote yields negative response
    response = adjust_response_style("Hello", state, signals=["negative", "negative"])
    assert response.startswith("I'm sorry to hear that.")

