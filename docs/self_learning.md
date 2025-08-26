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

The `SelfImprovement.evaluate_and_rollback` method checks the latest metrics
recorded in `evolution/metrics_history.csv`. When a metric falls below its
configured threshold the method executes a rollback script to restore the last
known good model.

## Monitoring and Alerts

Warnings emitted during failed deployments or rollbacks should be fed into the
system's monitoring and alerting pipeline. This enables operators to react
promptly when model quality regresses or automated rollbacks are triggered.

