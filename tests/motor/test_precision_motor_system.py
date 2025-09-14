import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules.brain.motor.precision import PrecisionMotorSystem


def test_planning_and_execution_flow():
    system = PrecisionMotorSystem()
    plan = system.plan_movement("reach target")
    assert "[BG]" in plan
    assert "optimized for obstacles and forces" in plan
    result = system.execute_action(plan)
    assert result.startswith("executed")
    assert "fine-tuned" in result
    assert "[BG]" in result


def test_cerebellar_learning_updates():
    system = PrecisionMotorSystem()
    system.learn("offset")
    plan = system.plan_movement("move")
    result = system.execute_action(plan)
    assert "offset" in result


def test_basal_ganglia_modulation():
    system = PrecisionMotorSystem()
    plan = system.plan_movement("raise hand")
    system.execute_action(plan)
    assert len(system.basal_ganglia.gating_history) == 2
