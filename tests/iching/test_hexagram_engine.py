import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from modules.iching.hexagram_engine import HexagramEngine


def test_interpretation_and_independence():
    engine = HexagramEngine()
    hexagram = "111111"
    dims = ["career", "relationship", "fortune"]

    result = engine.interpret_hexagram(hexagram, dims)

    assert result["career"] == {
        "interpretation": "Great potential for leadership",
        "advice": "Take charge of projects.",
    }
    assert result["relationship"] == {
        "interpretation": "Strong personal drive may overshadow others",
        "advice": "Practice empathy.",
    }
    assert result["fortune"] == {
        "interpretation": "Favorable circumstances ahead",
        "advice": "Capitalize on momentum.",
    }

    # Order of dimensions should not affect the outcome
    reversed_result = engine.interpret_hexagram(hexagram, list(reversed(dims)))
    assert result == reversed_result

    # Individual dimension interpretation should match subset of multi-dimensional result
    career_only = engine.interpret_hexagram(hexagram, ["career"])
    assert career_only["career"] == result["career"]


def test_different_hexagram_consistency():
    engine = HexagramEngine()
    hexagram = "000000"
    dims = ["career", "relationship", "fortune"]

    result = engine.interpret_hexagram(hexagram, dims)

    assert result["career"] == {
        "interpretation": "Need for support and collaboration",
        "advice": "Seek guidance from mentors.",
    }
    assert result["relationship"] == {
        "interpretation": "Stable and nurturing environment",
        "advice": "Foster open communication.",
    }
    assert result["fortune"] == {
        "interpretation": "Challenges may arise",
        "advice": "Remain patient and prepared.",
    }

    fortune_only = engine.interpret_hexagram(hexagram, ["fortune"])
    assert fortune_only["fortune"] == result["fortune"]
