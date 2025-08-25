# Retraining Pipeline

The `ml/retraining_pipeline.py` module automates model updates using new log
entries.

1. **Accumulate logs** – new logs are stored in `data/new_logs.csv` and are
   appended to the aggregated dataset `data/dataset.csv`.
2. **Retrain and evaluate** – the dataset is used to train a fresh model via
   `ml/train_models.py`. The resulting metrics are compared against the current
   baseline located in `artifacts/current/`.
3. **Deploy on improvement** – if the new model achieves a lower test MSE, it
   replaces the existing model in `artifacts/current/`.

To execute the pipeline periodically, use the provided `retrain_cron.sh` script
in a cron job or workflow:

```cron
0 0 * * * /path/to/retrain_cron.sh >> /path/to/retrain.log 2>&1
```

This runs the pipeline every day at midnight.
