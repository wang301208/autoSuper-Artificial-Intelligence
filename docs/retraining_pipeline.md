# Retraining Pipeline

The `ml/retraining_pipeline.py` module automates model updates using new log
entries. It now integrates a lightweight data pipeline for augmentation and
supports online learning via an event queue.

1. **Accumulate logs** – new logs are stored in `data/new_logs.csv` and are
   appended to the aggregated dataset `data/dataset.csv`. Before training the
   dataset passes through `ml/data_pipeline.py` where optional text and image
   augmentation/synthesis expands the data available for learning.
2. **Retrain and evaluate** – the dataset is used to train a fresh model.
   Classic models are trained via `ml/train_models.py` while LLMs use
   `ml/fine_tune_llm.py` when calling the pipeline with `--model llm`. The
   resulting metrics (MSE for classic models or perplexity for LLMs) are
   compared against the current baseline located in `artifacts/current/`.
3. **Deploy on improvement** – if the new model achieves a lower metric, it
   replaces the existing model in `artifacts/current/`. Older versions remain in
   `artifacts/<version>/` and can be restored manually to roll back.
4. **Online ingestion (optional)** – `backend/runner/streaming.py` exposes a
   queue based interface for feeding new samples from an event bus directly into
   an online learning loop. It can be coupled with the active learning sampler
   in `ml/active_sampler.py` to prioritise the most informative events.

To execute the pipeline periodically, use the provided `retrain_cron.sh` script
in a cron job or workflow. The script forwards all arguments to the Python
module, allowing LLM training:

```cron
0 0 * * * /path/to/retrain_cron.sh --model llm >> /path/to/retrain.log 2>&1
```

This runs the pipeline every day at midnight. To roll back to a previous model
version, copy the desired directory back to `artifacts/current/`:

```bash
rm -rf artifacts/current
cp -r artifacts/<old_version> artifacts/current
```
