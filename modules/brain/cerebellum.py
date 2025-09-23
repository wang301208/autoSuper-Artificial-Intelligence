"""Simplified cerebellum components for motor control and learning."""

from __future__ import annotations

from numbers import Real
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
        self.metric_history: List[Dict[str, float]] = []

    def fine_tune(self, motor_command: str | MotorCommand) -> str | MotorCommand:
        """Refine motor commands or command objects for smoother execution."""

        if isinstance(motor_command, MotorCommand):
            return self._fine_tune_command(motor_command)
        processed = self.granule.process(str(motor_command))
        return self.purkinje.refine(processed)

    def motor_learning(self, feedback: str | Dict[str, Any] | MotorExecutionResult) -> str:
        """Update internal state using explicit feedback signals."""

        record = self._normalise_feedback(feedback)
        metrics: Dict[str, float] | None = None
        telemetry = record.get("telemetry")
        if isinstance(telemetry, dict):
            metrics = {
                key: float(value)
                for key, value in telemetry.items()
                if isinstance(value, Real)
            }
        if record.get("success") is not None:
            metrics = metrics or {}
            metrics.setdefault("success_rate", 1.0 if record.get("success") else 0.0)
        if record.get("error"):
            metrics = metrics or {}
            metrics.setdefault("overall_error", 1.0)
        if metrics:
            self.update_feedback(metrics)
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
        adjustments: Dict[str, Any] = {
            "arguments": {"gain": gain, "damping": damping},
            "training_samples": count,
            "failure_samples": failures,
        }
        if self.metric_history:
            recent = self.metric_history[-min(len(self.metric_history), 5) :]
            if recent:
                avg_velocity = sum(m.get("velocity_error", 0.0) for m in recent) / len(recent)
                avg_stability = sum(m.get("stability_error", 0.0) for m in recent) / len(recent)
                avg_accuracy = sum(m.get("accuracy_error", 0.0) for m in recent) / len(recent)
                adjustments["arguments"].update(
                    {
                        "gain": max(0.5, adjustments["arguments"]["gain"] - avg_velocity * 0.3),
                        "damping": min(1.0, adjustments["arguments"]["damping"] + avg_stability * 0.4),
                        "accuracy_bias": max(0.0, min(1.0, 1.0 - avg_accuracy)),
                    }
                )
                adjustments["metric_samples"] = len(self.metric_history)
        return adjustments

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

    def update_feedback(self, metrics: Dict[str, Any]) -> None:
        """Integrate structured metric feedback for multi-channel learning."""

        numeric: Dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, Real):
                numeric[key] = float(value)
        if not numeric:
            return
        summary = {key: numeric[key] for key in sorted(numeric)}
        self.metric_history.append(summary)
        self.learned_signals.append({"label": "metrics", "metrics": summary, "success": summary.get("success_rate", 1.0) >= 0.5})


__all__ = ["Cerebellum", "GranuleNetwork", "PurkinjeNetwork"]
