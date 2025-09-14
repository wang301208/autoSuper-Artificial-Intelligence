import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.brain import VisualCortex, AuditoryCortex, SomatosensoryCortex


def test_visual_cortex():
    cortex = VisualCortex()
    result = cortex.process("image data")
    assert "edges" in result and "color" in result


def test_auditory_cortex():
    cortex = AuditoryCortex()
    result = cortex.process("audio data")
    assert "frequencies" in result and "interpretation" in result


def test_somatosensory_cortex():
    cortex = SomatosensoryCortex()
    result = cortex.process("stimulus")
    assert result == ["touch"]
