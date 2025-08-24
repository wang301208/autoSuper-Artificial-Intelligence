from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from evolution.founder import Founder


def test_founder_uses_model_predictions():
    founder = Founder()
    # Provide low current metrics to ensure suggestions come from predictions
    metrics = {"cpu_percent": 10, "memory_percent": 10}
    suggestions = founder._generate_suggestions(metrics)
    assert "CPU usage high; consider distributing tasks." in suggestions
    assert "Memory usage high; investigate memory leaks." in suggestions
