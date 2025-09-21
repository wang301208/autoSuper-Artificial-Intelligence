from __future__ import annotations

"""Motor cortex integrating cortical planning with actuators."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .motor.actions import (
    ActionMapping,
    ActuatorInterface,
    MotorCommand,
    MotorExecutionResult,
    MotorPlan,
)


class PrimaryMotor:
    """Primary motor cortex responsible for executing motor commands."""

    def execute(self, motor_command: str) -> str:
        return f"executed {motor_command}"


class PreMotorArea:
    """Pre-motor area that plans movements based on intention."""

    def plan(self, intention: str) -> str:
        return f"plan for {intention}"


class SupplementaryMotor:
    """Supplementary motor area that coordinates complex movements."""

    def organize(self, plan: str) -> str:
        return f"organized {plan}"


@dataclass
class MotorCortex:
    """Motor cortex orchestrating planning, ethics, and actuator dispatch."""

    basal_ganglia: Any = None
    cerebellum: Any = None
    spiking_backend: Any = None
    ethics: Any = None
    actuator: ActuatorInterface | None = None
    action_map: Dict[str, ActionMapping] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.primary_motor = PrimaryMotor()
        self.premotor_area = PreMotorArea()
        self.supplementary_motor = SupplementaryMotor()
        self.feedback_history: list[MotorExecutionResult] = []

    # ------------------------------------------------------------------
    # Planning ---------------------------------------------------------
    # ------------------------------------------------------------------
    def plan_movement(self, intention: str, parameters: Optional[Dict[str, Any]] = None) -> MotorPlan:
        """Plan a movement and generate a structured `MotorPlan`."""

        parameters = dict(parameters or {})
        first_stage = self.premotor_area.plan(intention)
        second_stage = self.supplementary_motor.organize(first_stage)
        stages = [first_stage, second_stage]
        final_stage = stages[-1]
        if self.basal_ganglia:
            final_stage = self.basal_ganglia.modulate(final_stage)
            stages.append(final_stage)

        plan = MotorPlan(intention=intention, stages=stages, parameters=parameters)
        plan.metadata["plan_summary"] = final_stage
        plan.command = self._map_plan_to_command(plan)
        return plan

    def register_action(self, intention: str, mapping: ActionMapping) -> None:
        """Register or override an action mapping for a given intention."""

        self.action_map[intention] = mapping

    # ------------------------------------------------------------------
    # Execution --------------------------------------------------------
    # ------------------------------------------------------------------
    def execute_action(self, motor_instruction: Any):
        """Execute motor instructions via the actuator or primary motor."""

        # Neuromorphic execution path remains untouched for compatibility.
        if self.spiking_backend and isinstance(motor_instruction, (list, tuple)):
            outputs = self.spiking_backend.run(motor_instruction)
            self.spiking_backend.synapses.adapt(
                self.spiking_backend.spike_times, self.spiking_backend.spike_times
            )
            return outputs

        plan, command = self._ensure_command(motor_instruction)

        if self.ethics:
            report = self.ethics.evaluate_action(command.operation)
            if not report.get("compliant", True):
                return report

        tuned_command, textual_hint = self._apply_cerebellum(command, plan)

        if self.actuator:
            result = self.actuator.execute(tuned_command)
            self._post_execute_feedback(result)
            return result

        command_text = textual_hint or tuned_command.metadata.get("plan_summary", tuned_command.operation)
        return self.primary_motor.execute(str(command_text))

    # ------------------------------------------------------------------
    # Learning ---------------------------------------------------------
    # ------------------------------------------------------------------
    def train(self, feedback: MotorExecutionResult | str | Dict[str, Any]) -> str:
        """Update cerebellar model with rich feedback information."""

        if not self.cerebellum:
            return f"no cerebellum to learn from {feedback}"
        return self.cerebellum.motor_learning(feedback)

    # ------------------------------------------------------------------
    # Internal helpers -------------------------------------------------
    # ------------------------------------------------------------------
    def _map_plan_to_command(self, plan: MotorPlan) -> MotorCommand:
        mapping = self.action_map.get(plan.intention)
        if mapping:
            command = mapping.to_command(plan)
        else:
            metadata = {
                "plan_summary": plan.describe(),
                "stages": list(plan.stages),
                "intention": plan.intention,
            }
            command = MotorCommand("motor", plan.intention, dict(plan.parameters), metadata)
        command.metadata.setdefault("plan_summary", plan.describe())
        return command

    def _ensure_command(self, motor_instruction: Any) -> Tuple[MotorPlan | None, MotorCommand]:
        if isinstance(motor_instruction, MotorPlan):
            command = motor_instruction.command or self._map_plan_to_command(motor_instruction)
            return motor_instruction, command
        if isinstance(motor_instruction, MotorCommand):
            return None, motor_instruction
        if isinstance(motor_instruction, dict) and "operation" in motor_instruction:
            return None, MotorCommand(
                tool=motor_instruction.get("tool", "motor"),
                operation=motor_instruction["operation"],
                arguments=dict(motor_instruction.get("arguments", {})),
                metadata=dict(motor_instruction.get("metadata", {})),
            )
        if isinstance(motor_instruction, str):
            plan = self.plan_movement(motor_instruction)
            return plan, plan.command  # type: ignore[return-value]
        raise TypeError(f"Unsupported motor instruction type: {type(motor_instruction)!r}")

    def _apply_cerebellum(
        self, command: MotorCommand, plan: MotorPlan | None
    ) -> Tuple[MotorCommand, Optional[str]]:
        if not self.cerebellum:
            return command, plan.describe() if plan else None
        tuned_command = self.cerebellum.fine_tune(command)
        textual_hint = None
        if plan:
            textual_hint = self.cerebellum.fine_tune(plan.describe())
            if isinstance(textual_hint, MotorCommand):
                textual_hint = textual_hint.metadata.get("plan_summary", plan.describe())
        if isinstance(tuned_command, MotorCommand):
            return tuned_command, textual_hint
        return command, str(tuned_command)

    def _post_execute_feedback(self, result: MotorExecutionResult) -> None:
        self.feedback_history.append(result)
        if self.cerebellum:
            self.cerebellum.motor_learning(result)


__all__ = ["MotorCortex", "PrimaryMotor", "PreMotorArea", "SupplementaryMotor"]
