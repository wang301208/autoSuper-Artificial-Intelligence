import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.getcwd()))

from modules.brain.multimodal import CrossModalTransformer, MultimodalFusionEngine


def test_fused_representation_shape_and_value():
    transformer = CrossModalTransformer(output_dim=4)
    engine = MultimodalFusionEngine(transformer)
    visual = np.array([1.0, 3.0])
    auditory = np.array([2.0, 2.0])
    tactile = np.array([1.0, 1.0, 1.0])
    fused = engine.fuse_sensory_modalities(visual, auditory, tactile)
    expected_val = np.mean([visual.mean(), auditory.mean(), tactile.mean()])
    assert fused.shape == (4,)
    np.testing.assert_allclose(fused, expected_val)


def test_missing_modality_raises_error():
    engine = MultimodalFusionEngine()
    visual = np.array([1.0])
    auditory = np.array([1.0])
    with pytest.raises(ValueError):
        engine.fuse_sensory_modalities(visual, auditory, None)


def test_dependency_injection():
    class DummyTransformer:
        def fuse(self, modalities):
            return np.array([sum(np.sum(m) for m in modalities)])

    engine = MultimodalFusionEngine()
    engine.set_transformer(DummyTransformer())
    fused = engine.fuse_sensory_modalities(
        np.array([1.0]), np.array([2.0]), np.array([3.0])
    )
    assert fused.shape == (1,)
    assert fused[0] == 6.0
