# Performance Monitor Anomaly Detection

The `PerformanceMonitor` can automatically learn baseline performance characteristics and flag abnormal behavior using an Isolation Forest model.

## Configuration

Enable anomaly detection when constructing the monitor:

```python
from monitoring import PerformanceMonitor, TimeSeriesStorage

storage = TimeSeriesStorage()
monitor = PerformanceMonitor(
    storage,
    training_accuracy=0.95,
    degradation_threshold=0.05,
    enable_anomaly_detection=True,
    model_update_interval=3600,  # seconds between model retraining
    contamination=0.05,          # expected proportion of anomalies
)
```

### Parameters

- **enable_anomaly_detection** – turn on automatic anomaly detection.
- **model_update_interval** – seconds between model retraining on recent metrics.
- **contamination** – estimated fraction of anomalous samples in the data.

The model uses CPU and memory usage events stored in `TimeSeriesStorage` as input features and periodically retrains to adapt to changing baselines.

## Alerts

When the detector flags an anomalous deviation, the monitor triggers configured alert handlers. The alert message includes the component with the highest failure count returned by `TimeSeriesStorage.bottlenecks()` to hint at the most likely bottleneck.

### Resource thresholds

In addition to anomaly detection, `PerformanceMonitor` can enforce hard
resource limits.  Configure thresholds when constructing the monitor to
emit alerts if average usage crosses the limits:

```python
monitor = PerformanceMonitor(
    storage,
    training_accuracy=0.95,
    degradation_threshold=0.05,
    cpu_threshold=80.0,        # percent
    memory_threshold=85.0,     # percent
    throughput_threshold=5.0,  # tasks/sec minimum
    alert_handlers=[dashboard_alert()],
)
```

### Alert handlers

Alert handlers determine how notifications are delivered.  Two helpers are
provided:

```python
from monitoring import email_alert, dashboard_alert

handlers = [
    email_alert("ops@example.com"),  # send emails
    dashboard_alert(),               # log for dashboards
]
monitor = PerformanceMonitor(storage, 1.0, 0.1, alert_handlers=handlers)
```

Email alerts use SMTP to send notifications, while the dashboard handler
simply logs the message for ingestion by observability systems.
