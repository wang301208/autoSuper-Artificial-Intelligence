import sys
from pathlib import Path

import pytest

# Ensure repository root is on the import path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from modules.iching import YinYangFiveElements


def test_transform_yinyang():
    assert YinYangFiveElements.transform_yinyang("yin") == "yang"
    assert YinYangFiveElements.transform_yinyang("yang") == "yin"
    with pytest.raises(ValueError):
        YinYangFiveElements.transform_yinyang("none")


def test_element_interaction_generates():
    assert (
        YinYangFiveElements.element_interaction("wood", "fire") == "generates"
    )
    assert (
        YinYangFiveElements.element_interaction("fire", "wood") == "generated_by"
    )


def test_element_interaction_overcomes():
    assert (
        YinYangFiveElements.element_interaction("wood", "earth") == "overcomes"
    )
    assert (
        YinYangFiveElements.element_interaction("earth", "wood") == "overcome_by"
    )


def test_element_interaction_neutral_and_overcome_by():
    assert (
        YinYangFiveElements.element_interaction("wood", "metal") == "overcome_by"
    )
    assert (
        YinYangFiveElements.element_interaction("wood", "wood") == "neutral"
    )
