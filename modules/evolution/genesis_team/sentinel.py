"""Sentinel agent monitors for anomalies."""

from __future__ import annotations

from .. import Agent


class Sentinel(Agent):
    """Monitors system state for anomalies."""

    def perform(self) -> str:
        # Placeholder monitoring logic
        return "Sentinel monitoring complete; no anomalies detected."
