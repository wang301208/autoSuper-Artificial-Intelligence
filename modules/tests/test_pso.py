import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402
from pso import pso  # noqa: E402


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


def test_pso_finds_minimum():
    bounds = [(-5, 5)] * 3
    result = pso(sphere, bounds, num_particles=15, max_iter=80)
    assert result.value < 1e-2
    assert np.allclose(result.position, np.zeros(3), atol=1e-1)
