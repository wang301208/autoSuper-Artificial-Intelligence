"""Simplified cerebellum components for motor control and learning."""

from __future__ import annotations

from typing import Any, Dict, List

from .motor.actions import MotorCommand, MotorExecutionResult


class PurkinjeNetwork:
    """Network of Purkinje cells for refining motor signals."""

    def refine(self, signal: str) -> str:
        """Refine incoming signals from granule network."""
        return f"{signal} refined"


class GranuleNetwork:
    """Network of granule cells preprocessing inputs."""

    def process(self, input_signal: str) -> str:
        """Process incoming sensory or motor inputs."""
        return f"processed {input_signal}"


class Cerebellum:
    """Simplified cerebellum coordinating motor control and learning."""

    def __init__(self) -> None:
        self.purkinje = PurkinjeNetwork()
        self.granule = GranuleNetwork()
        self.learned_signals: List[Dict[str, Any]] = []

    def fine_tune(self, motor_command: str | MotorCommand) -> str | MotorCommand:
        """Refine motor commands or command objects for smoother execution."""

        if isinstance(motor_command, MotorCommand):
            return self._fine_tune_command(motor_command)
        processed = self.granule.process(str(motor_command))
        return self.purkinje.refine(processed)

    def motor_learning(self, feedback: str | Dict[str, Any] | MotorExecutionResult) -> str:
        """Update internal state using explicit feedback signals."""

        record = self._normalise_feedback(feedback)
        self.learned_signals.append(record)
        label = record.get("label", record.get("signal", "feedback"))
        return f"learned from {label}"

    def balance_control(self, sensory_input: str) -> str:
        """Placeholder for balance control using sensory feedback."""

        processed = self.granule.process(sensory_input)
        return f"balance adjusted with {processed}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _fine_tune_command(self, command: MotorCommand) -> MotorCommand:
        adjustments = self._compute_adjustments()
        metadata = {"cerebellum": adjustments}
        argument_updates = adjustments.get("arguments", {})
        return command.with_updates(arguments=argument_updates, metadata=metadata)

    def _compute_adjustments(self) -> Dict[str, Any]:
        count = len(self.learned_signals)
        if count == 0:
            return {"arguments": {}, "training_samples": 0}
        failures = sum(1 for rec in self.learned_signals if not rec.get("success", True))
        gain = 1.0 + min(count * 0.05, 0.5)
        damping = min(failures * 0.1, 0.6)
        return {
            "arguments": {"gain": gain, "damping": damping},
            "training_samples": count,
            "failure_samples": failures,
        }

    def _normalise_feedback(self, feedback: str | Dict[str, Any] | MotorExecutionResult) -> Dict[str, Any]:
        if isinstance(feedback, str):
            return {"signal": feedback, "label": feedback, "success": False}
        if isinstance(feedback, dict):
            data = dict(feedback)
            data.setdefault("label", str(data.get("signal", "feedback")))
            return data
        return {
            "signal": feedback.error or "execution",
            "label": feedback.error or ("success" if feedback.success else "execution"),
            "success": feedback.success,
            "telemetry": dict(feedback.telemetry),
        }


__all__ = ["Cerebellum", "GranuleNetwork", "PurkinjeNetwork"]
