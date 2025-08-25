import sys, os
sys.path.insert(0, os.path.abspath(os.getcwd()))

from pathlib import Path

from monitoring import PerformanceMonitor, TimeSeriesStorage


def test_performance_monitor_resource_alert(tmp_path: Path) -> None:
    storage = TimeSeriesStorage(tmp_path / "monitoring.db")
    alerts: list[tuple[str, str]] = []

    def handler(subj: str, msg: str) -> None:
        alerts.append((subj, msg))

    monitor = PerformanceMonitor(
        storage,
        training_accuracy=1.0,
        degradation_threshold=0.1,
        alert_handlers=[handler],
        cpu_threshold=50.0,
    )
    monitor.log_resource_usage("agent", 90.0, 10.0)
    monitor.check_performance()
    assert any("CPU" in a[0] for a in alerts)
