# Self Learning Pipeline

This document outlines how AutoGPT evaluates new models, performs rollbacks,
and monitors performance.

## Evaluation Metrics

During retraining the system compares a set of metrics between the newly
trained model and the current baseline. Typical metrics include success rate,
mean reward, and the core training metric (e.g. perplexity or mean squared
error). Each metric has an associated threshold that defines the minimum
acceptable improvement. If any metric fails to meet its threshold, deployment
is skipped and a warning is logged.

## Rollback Strategy

Metrics can be recorded in CSV, JSON, or YAML format. Nested structures are
flattened so that downstream logic receives simple key/value pairs. The
`SelfImprovement.evaluate_and_rollback` method checks the latest metrics
recorded in `evolution/metrics_history.*`. When a metric falls below its
configured threshold the method executes a rollback script to restore the last
known good model. The genetic algorithm history file (`ga_metrics_history.*`)
supports the same formats and will emit JSON when the filename ends with
`.json`.

## Monitoring and Alerts

Warnings emitted during failed deployments or rollbacks should be fed into the
system's monitoring and alerting pipeline. This enables operators to react
promptly when model quality regresses or automated rollbacks are triggered.

## Automatic Scheduling

AutoGPT can automatically run the retraining pipeline and self-improvement
routine at fixed intervals. When an agent is started via
`python cli.py agent start <name>`, a background scheduler is launched.

Intervals are configured through environment variables (values are in seconds):

| Variable | Purpose |
| --- | --- |
| `AUTO_RETRAIN_INTERVAL` | Run `ml.retraining_pipeline.main` periodically |
| `AUTO_SELF_IMPROVE_INTERVAL` | Invoke `SelfImprovement.run` periodically |
| `AUTO_SCHEDULE_INTERVAL` | Fallback interval if the above are unset |

If none of these variables are set to a positive value the scheduler exits
immediately.

Example:

```bash
export AUTO_RETRAIN_INTERVAL=3600           # retrain hourly
export AUTO_SELF_IMPROVE_INTERVAL=86400     # self-improve daily
python cli.py agent start forge
```

