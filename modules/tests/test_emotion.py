import os
import sys

sys.path.insert(0, os.path.abspath(os.getcwd()))

from modules.common.emotion import EmotionAnalyzer, EmotionState, adjust_response_style


def test_emotion_analyzer_classifies_basic_sentiment():
    analyzer = EmotionAnalyzer()
    assert analyzer.analyze_text("I love this!") == "positive"
    assert analyzer.analyze_text("This is terrible") == "negative"
    assert analyzer.analyze_text("It is a table") == "neutral"


def test_emotion_state_updates_and_tracks_variables():
    analyzer = EmotionAnalyzer()
    state = EmotionState()
    state.update("I love this!", analyzer)
    assert state.label == "positive"
    positive_excitement = state.excitement
    state.update("I hate that", analyzer)
    assert state.label == "negative"
    assert state.excitement < positive_excitement


def test_adjust_response_style_reflects_emotion():
    analyzer = EmotionAnalyzer()
    state = EmotionState()
    state.update("I love this", analyzer)
    response = adjust_response_style("Thanks for the feedback.", state)
    assert "😊" in response
    state.update("I hate that", analyzer)
    response = adjust_response_style("Thanks for the feedback.", state)
    assert response.startswith("I'm sorry to hear that.")
