import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.brain import MotorCortex, Cerebellum


def test_cerebellum_training_integration():
    cerebellum = Cerebellum()
    cortex = MotorCortex(cerebellum=cerebellum)
    result = cortex.train("overshoot")
    assert "learned" in result and "overshoot" in result
    executed = cortex.execute_action("move")
    assert "refined" in executed


def test_balance_control_placeholder():
    cerebellum = Cerebellum()
    response = cerebellum.balance_control("tilt")
    assert "balance adjusted" in response and "tilt" in response
