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
