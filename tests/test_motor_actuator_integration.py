import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.brain import MotorCortex, Cerebellum
from modules.brain.motor.actions import ActionMapping, CallableActuator, MotorCommand, MotorExecutionResult


def test_motor_cortex_actuator_mapping_and_feedback():
    dispatched: dict[str, MotorCommand] = {}

    def dispatcher(command: MotorCommand) -> MotorExecutionResult:
        dispatched["command"] = command
        return MotorExecutionResult(success=True, output="ok", telemetry={"latency": 0.01})

    actuator = CallableActuator(dispatcher)
    cerebellum = Cerebellum()
    cortex = MotorCortex(cerebellum=cerebellum, actuator=actuator)
    cortex.register_action(
        "reach", ActionMapping(tool="manipulator", operation="move_arm", argument_defaults={"speed": 0.5})
    )

    cortex.train("overshoot")
    plan = cortex.plan_movement("reach", {"target": [0.2, 0.3, 0.4]})
    result = cortex.execute_action(plan)

    assert isinstance(result, MotorExecutionResult)
    assert result.success
    assert "command" in dispatched
    command = dispatched["command"]
    assert command.operation == "move_arm"
    assert command.arguments["target"] == [0.2, 0.3, 0.4]
    assert command.metadata.get("cerebellum", {}).get("training_samples") >= 1
    assert cerebellum.metric_history
    assert any("velocity_error" in entry for entry in cerebellum.metric_history)
    assert any("stability_error" in entry for entry in cerebellum.metric_history)


def test_motor_cortex_train_with_execution_result():
    cortex = MotorCortex(cerebellum=Cerebellum())
    feedback = MotorExecutionResult(success=False, output=None, telemetry={"latency": 0.2}, error="collision")
    report = cortex.train(feedback)
    assert "learned" in report and "collision" in report
    assert cortex.cerebellum.metric_history
    metrics = cortex.cerebellum.metric_history[-1]
    assert metrics["success_rate"] < 1.0
