# Self Learning Pipeline

This document outlines the metrics and safeguards used when the system trains
new models autonomously.

## Evaluation metrics

`ml/retraining_pipeline.deploy_if_better` compares the new model against the
currently deployed one using several metrics:

- **success_rate** – fraction of successful executions.
- **reward_mean** – average reward produced by the model.
- **test_mse** – mean squared error on held‑out data.

Each metric has a threshold. The new model must improve `success_rate` by at
least 1 percentage point, keep `reward_mean` from decreasing, and reduce
`test_mse`. Metrics are read from `artifacts/<version>/metrics.txt`.

If any metric fails to meet its threshold the deployment is aborted and a
warning is logged, preserving the existing `current` model.

## Rollback strategy

`evolution.self_improvement.SelfImprovement.evaluate_and_rollback()` reads the
latest entries from `evolution/metrics_history.csv`. When metrics drop below
predefined minima it executes a rollback script (default
`scripts/rollback.sh`) to restore a previous stable state.

## Monitoring and alerts

- Retraining runs periodically and records metrics for each model version.
- The self‑improvement module monitors these metrics and emits warnings when
  they regress.
- When a rollback is triggered the event should be surfaced to operators via
  standard logging/alerting tools.
