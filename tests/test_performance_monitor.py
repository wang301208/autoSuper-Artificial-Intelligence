import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from monitoring.performance_monitor import PerformanceMonitor
from monitoring.storage import TimeSeriesStorage


def test_alert_triggered_on_accuracy_drop(tmp_path):
    storage = TimeSeriesStorage(tmp_path / "perf.db")
    alerted = []

    def handler(subject: str, message: str) -> None:
        alerted.append((subject, message))

    monitor = PerformanceMonitor(
        storage,
        training_accuracy=0.9,
        degradation_threshold=0.1,
        alert_handlers=[handler],
    )

    monitor.log_prediction(1, 1)
    monitor.log_prediction(1, 0)
    monitor.log_prediction(0, 1)
    monitor.check_performance()

    assert alerted, "Performance drop should trigger alert"
    events = storage.events("prediction")
    assert len(events) == 3
