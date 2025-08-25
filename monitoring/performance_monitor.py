"""Utilities for tracking model performance in production."""

from __future__ import annotations

from email.message import EmailMessage
from typing import Any, Callable, Iterable, List
import smtplib

from .storage import TimeSeriesStorage

AlertHandler = Callable[[str, str], None]


class PerformanceMonitor:
    """Monitor predictions against outcomes and trigger alerts on degradation."""

    def __init__(
        self,
        storage: TimeSeriesStorage,
        training_accuracy: float,
        degradation_threshold: float,
        alert_handlers: Iterable[AlertHandler] | None = None,
    ) -> None:
        self.storage = storage
        self.training_accuracy = training_accuracy
        self.degradation_threshold = degradation_threshold
        self.alert_handlers: List[AlertHandler] = list(alert_handlers or [])

    # ------------------------------------------------------------------
    # logging
    # ------------------------------------------------------------------
    def log_prediction(self, prediction: Any, outcome: Any) -> None:
        """Persist *prediction* and corresponding *outcome*."""
        status = "success" if prediction == outcome else "failure"
        event = {"prediction": prediction, "outcome": outcome, "status": status}
        self.storage.store("prediction", event)

    # ------------------------------------------------------------------
    # metrics
    # ------------------------------------------------------------------
    def current_accuracy(self) -> float:
        """Return accuracy computed from logged predictions."""
        events = self.storage.events("prediction")
        if not events:
            return 0.0
        correct = sum(1 for e in events if e.get("prediction") == e.get("outcome"))
        return correct / len(events)

    def check_performance(self) -> None:
        """Compare live accuracy against training benchmark and alert if degraded."""
        accuracy = self.current_accuracy()
        allowed_drop = self.training_accuracy - self.degradation_threshold
        if accuracy < allowed_drop:
            self._alert(
                "Model performance degraded",
                f"Accuracy {accuracy:.2%} below threshold {allowed_drop:.2%}",
            )

    # ------------------------------------------------------------------
    # alerting
    # ------------------------------------------------------------------
    def _alert(self, subject: str, message: str) -> None:
        for handler in self.alert_handlers:
            try:
                handler(subject, message)
            except Exception:
                pass


def email_alert(
    to_address: str,
    smtp_server: str = "localhost",
    smtp_port: int = 25,
    from_address: str = "noreply@example.com",
) -> AlertHandler:
    """Return an alert handler that sends email notifications."""

    def send(subject: str, message: str) -> None:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = from_address
        msg["To"] = to_address
        msg.set_content(message)
        with smtplib.SMTP(smtp_server, smtp_port) as smtp:
            smtp.send_message(msg)

    return send


def dashboard_alert(logger: Callable[[str], None] = print) -> AlertHandler:
    """Return an alert handler that logs notifications for dashboards."""

    def send(subject: str, message: str) -> None:
        logger(f"{subject}: {message}")

    return send
