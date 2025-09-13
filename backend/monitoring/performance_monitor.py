"""Utilities for tracking model performance in production."""

from __future__ import annotations

import time
from email.message import EmailMessage
from typing import Any, Callable, Iterable, List
import smtplib

from sklearn.ensemble import IsolationForest

from .storage import TimeSeriesStorage
from common import AutoGPTException, log_and_format_exception
from smtplib import SMTPException

AlertHandler = Callable[[str, str], None]


class PerformanceMonitor:
    """Monitor predictions against outcomes and trigger alerts on degradation."""

    def __init__(
        self,
        storage: TimeSeriesStorage,
        training_accuracy: float,
        degradation_threshold: float,
        alert_handlers: Iterable[AlertHandler] | None = None,
        cpu_threshold: float | None = None,
        memory_threshold: float | None = None,
        throughput_threshold: float | None = None,
        *,
        enable_anomaly_detection: bool = False,
        model_update_interval: float = 3600.0,
        contamination: float = 0.05,
    ) -> None:
        self.storage = storage
        self.training_accuracy = training_accuracy
        self.degradation_threshold = degradation_threshold
        self.alert_handlers: List[AlertHandler] = list(alert_handlers or [])
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.throughput_threshold = throughput_threshold
        self.enable_anomaly_detection = enable_anomaly_detection
        self.model_update_interval = model_update_interval
        self.contamination = contamination
        self._anomaly_model: IsolationForest | None = None
        self._last_model_update = 0.0

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

    # ------------------------------------------------------------------
    # resource metrics
    # ------------------------------------------------------------------
    def log_resource_usage(self, agent: str, cpu: float, memory: float) -> None:
        """Persist resource usage for *agent*."""
        self.storage.store("agent.resource", {"agent": agent, "cpu": cpu, "memory": memory})

    def log_task_completion(self, agent: str) -> None:
        """Record completion of a task by *agent*."""
        self.storage.store("task", {"agent": agent})

    def cpu_usage(self, agent: str | None = None, interval: float = 60.0) -> float:
        """Return average CPU usage for *agent* over *interval* seconds."""
        start = time.time() - interval
        events = self.storage.events("agent.resource", start_ts=start)
        samples = [e for e in events if agent is None or e.get("agent") == agent]
        if not samples:
            return 0.0
        return sum(float(e.get("cpu", 0.0)) for e in samples) / len(samples)

    def memory_usage(self, agent: str | None = None, interval: float = 60.0) -> float:
        """Return average memory usage for *agent* over *interval* seconds."""
        start = time.time() - interval
        events = self.storage.events("agent.resource", start_ts=start)
        samples = [e for e in events if agent is None or e.get("agent") == agent]
        if not samples:
            return 0.0
        return sum(float(e.get("memory", 0.0)) for e in samples) / len(samples)

    def task_throughput(self, agent: str | None = None, interval: float = 60.0) -> float:
        """Return tasks completed per second for *agent* over *interval* seconds."""
        start = time.time() - interval
        events = self.storage.events("task", start_ts=start)
        count = sum(1 for e in events if agent is None or e.get("agent") == agent)
        return count / interval if interval > 0 else 0.0

    def _resource_samples(self) -> list[list[float]]:
        events = self.storage.events("agent.resource")
        return [
            [float(e.get("cpu", 0.0)), float(e.get("memory", 0.0))]
            for e in events
        ]

    def _latest_resource_sample(self) -> list[float] | None:
        events = self.storage.events("agent.resource")
        if events:
            e = events[-1]
            return [float(e.get("cpu", 0.0)), float(e.get("memory", 0.0))]
        return None

    def _maybe_update_model(self) -> None:
        now = time.time()
        if self._anomaly_model is None or now - self._last_model_update > self.model_update_interval:
            data = self._resource_samples()
            if data:
                self._anomaly_model = IsolationForest(contamination=self.contamination)
                self._anomaly_model.fit(data)
                self._last_model_update = now

    def check_performance(self) -> None:
        """Compare live metrics against thresholds and alert on degradation."""
        accuracy = self.current_accuracy()
        allowed_drop = self.training_accuracy - self.degradation_threshold
        if accuracy < allowed_drop:
            self._alert(
                "Model performance degraded",
                f"Accuracy {accuracy:.2%} below threshold {allowed_drop:.2%}",
            )

        # Check resource utilization
        if self.cpu_threshold is not None:
            cpu = self.cpu_usage()
            if cpu > self.cpu_threshold:
                self._alert(
                    "CPU usage high",
                    f"Average CPU usage {cpu:.2f}% exceeds {self.cpu_threshold:.2f}%",
                )
        if self.memory_threshold is not None:
            mem = self.memory_usage()
            if mem > self.memory_threshold:
                self._alert(
                    "Memory usage high",
                    f"Average memory usage {mem:.2f}% exceeds {self.memory_threshold:.2f}%",
                )

        if self.throughput_threshold is not None:
            tp = self.task_throughput()
            if tp < self.throughput_threshold:
                self._alert(
                    "Task throughput low",
                    f"Throughput {tp:.2f} tasks/sec below {self.throughput_threshold:.2f}",
                )

        if self.enable_anomaly_detection:
            self._maybe_update_model()
            sample = self._latest_resource_sample()
            if sample and self._anomaly_model is not None:
                pred = self._anomaly_model.predict([sample])[0]
                if pred == -1:
                    bottlenecks = self.storage.bottlenecks()
                    component = max(bottlenecks, key=bottlenecks.get) if bottlenecks else "unknown"
                    self._alert(
                        "Anomalous performance metrics",
                        f"Metrics {sample} flagged as anomalous. Potential bottleneck: {component}",
                    )


    # ------------------------------------------------------------------
    # alerting
    # ------------------------------------------------------------------
    def _alert(self, subject: str, message: str) -> None:
        for handler in self.alert_handlers:
            try:
                handler(subject, message)
            except AutoGPTException as err:
                log_and_format_exception(err)
            except SMTPException as err:
                log_and_format_exception(err)
            except Exception as err:  # pragma: no cover - unexpected
                log_and_format_exception(err)


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
