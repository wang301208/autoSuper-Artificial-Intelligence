import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.brain import MotorCortex


class StubBasalGanglia:
    def modulate(self, plan: str) -> str:
        return plan + " modulated"


class StubCerebellum:
    def fine_tune(self, command: str) -> str:
        return command + " tuned"


def test_motor_cortex_plan_execute():
    cortex = MotorCortex(basal_ganglia=StubBasalGanglia(), cerebellum=StubCerebellum())
    plan = cortex.plan_movement("wave")
    assert "modulated" in plan
    result = cortex.execute_action(plan)
    assert "executed" in result and "tuned" in result
