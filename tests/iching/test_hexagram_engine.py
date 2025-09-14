import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from modules.iching.hexagram_engine import HexagramEngine


def test_predict_transformations_with_changes():
    engine = HexagramEngine([0, 0, 0, 0, 0, 0])
    result = engine.predict_transformations(changes=[[0, 5], [2]])
    assert result.path == [
        (0, 0, 0, 0, 0, 0),
        (1, 0, 0, 0, 0, 1),
        (1, 0, 1, 0, 0, 1),
    ]
    assert result.yang_counts == [0, 2, 3]
    assert result.trend == "increasing"
    assert "increasing" in result.report


def test_predict_transformations_with_steps():
    engine = HexagramEngine([1, 1, 1, 1, 1, 1])
    result = engine.predict_transformations(steps=2)
    assert result.path == [
        (1, 1, 1, 1, 1, 1),
        (0, 1, 1, 1, 1, 1),
        (0, 0, 1, 1, 1, 1),
    ]
    assert result.yang_counts == [6, 5, 4]
    assert result.trend == "decreasing"
    assert "decreasing" in result.report
