from __future__ import annotations

"""Simple API for exposing stored monitoring metrics."""

from fastapi import FastAPI

from .storage import TimeSeriesStorage


def create_app(storage: TimeSeriesStorage | None = None) -> FastAPI:
    storage = storage or TimeSeriesStorage()
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

    return app
