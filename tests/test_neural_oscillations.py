import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.brain import NeuralOscillations


def test_synchronize_regions():
    oscillations = NeuralOscillations()
    regions = ["hippocampus", "cortex"]
    result = oscillations.synchronize_regions(regions)
    expected = [f"{r} synchronized via gamma waves" for r in regions]
    assert result == expected
