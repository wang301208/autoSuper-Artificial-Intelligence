from __future__ import annotations

"""Simple API for exposing stored monitoring metrics."""

from fastapi import FastAPI

from .storage import TimeSeriesStorage
from .evaluation import EvaluationMetrics


def create_app(
    storage: TimeSeriesStorage | None = None,
    evaluation: EvaluationMetrics | None = None,
) -> FastAPI:
    storage = storage or TimeSeriesStorage()
    evaluation = evaluation or EvaluationMetrics()
    app = FastAPI()

    @app.get("/metrics/{topic}")
    def get_events(topic: str, limit: int = 100):
        """Return recent events for *topic*."""
        return storage.events(topic, limit=limit)

    @app.get("/metrics/summary")
    def summary():
        """Return aggregated performance metrics."""
        return {
            "success_rate": storage.success_rate(),
            "bottlenecks": storage.bottlenecks(),
            "blueprint_versions": storage.blueprint_versions(),
        }

    @app.get("/metrics/evaluation")
    def evaluation_summary():
        """Return precision/recall, latency and fairness metrics."""
        return evaluation.summary()

    @app.get("/metrics/explanations")
    def explanations(limit: int = 100):
        """Return logged model explanations."""
        return storage.events("analysis.explanations", limit=limit)

    return app
