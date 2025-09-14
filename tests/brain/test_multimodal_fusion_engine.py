import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.multimodal.fusion_engine import MultimodalFusionEngine
from modules.brain.multimodal.cross_modal_transformer import CrossModalTransformer


def _manual_fusion(*modalities):
    trans = CrossModalTransformer()
    aligned = [trans.project(m) for m in modalities]
    scores = np.array([np.linalg.norm(a) for a in aligned], dtype=float)
    if np.allclose(scores, 0):
        weights = np.full(len(aligned), 1.0 / len(aligned))
    else:
        scores -= np.max(scores)
        exp_scores = np.exp(scores)
        weights = exp_scores / exp_scores.sum()
    return np.average(aligned, axis=0, weights=weights)


def test_attention_weighting():
    engine = MultimodalFusionEngine()
    visual = np.ones((2, 2))  # mean=1
    auditory = np.full((2, 2), 10)  # mean=10
    fused = engine.fuse_sensory_modalities(visual=visual, auditory=auditory)
    expected = _manual_fusion(visual, auditory)
    assert np.allclose(fused, expected)


def test_support_additional_modalities():
    engine = MultimodalFusionEngine()
    modalities = {
        "visual": np.array([1, 2, 3]),
        "auditory": np.array([4, 5, 6]),
        "tactile": np.array([7, 8, 9]),
        "smell": np.array([1, 1, 1]),
        "text": np.array([10, 10, 10]),
    }
    fused = engine.fuse_sensory_modalities(**modalities)
    expected = _manual_fusion(*modalities.values())
    assert np.allclose(fused, expected)


def test_robustness_zero_input():
    engine = MultimodalFusionEngine()
    zero = np.zeros(5)
    one = np.ones(5)
    fused = engine.fuse_sensory_modalities(zero=zero, one=one)
    expected = _manual_fusion(zero, one)
    assert np.allclose(fused, expected)
