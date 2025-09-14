import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from iching.ai_interpreter import AIEnhancedInterpreter
from iching.hexagram_engine import HexagramEngine


def test_same_hexagram_different_contexts():
    interpreter = AIEnhancedInterpreter()
    engine = HexagramEngine(interpreter)

    career_interp = engine.get_interpretation("乾", context="career")
    relationship_interp = engine.get_interpretation("乾", context="relationships")

    base = interpreter.knowledge_base["乾"]

    assert base in career_interp
    assert base in relationship_interp
    assert career_interp != relationship_interp
