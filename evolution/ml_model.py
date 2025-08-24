"""Simple machine learning model for predicting system resource usage."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.linear_model import LinearRegression


class ResourceModel:
    """Train models on historical metrics and predict future usage."""

    def __init__(self, data_path: Path | str | None = None) -> None:
        self.data_path = (
            Path(data_path)
            if data_path is not None
            else Path(__file__).with_name("metrics_history.csv")
        )
        self.cpu_model = LinearRegression()
        self.mem_model = LinearRegression()
        self._trained = False

    def _load(self) -> np.ndarray:
        data: list[tuple[float, float]] = []
        try:
            with open(self.data_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append(
                        (
                            float(row["cpu_percent"]),
                            float(row["memory_percent"]),
                        )
                    )
        except FileNotFoundError:
            return np.empty((0, 2))
        return np.array(data)

    def train(self) -> None:
        """Train regression models on historical data."""
        data = self._load()
        if data.size == 0:
            self._trained = False
            return
        indices = np.arange(len(data)).reshape(-1, 1)
        self.cpu_model.fit(indices, data[:, 0])
        self.mem_model.fit(indices, data[:, 1])
        self._trained = True

    def predict_next(self) -> Dict[str, float]:
        """Predict next CPU and memory usage values."""
        if not self._trained:
            self.train()
        if not self._trained:
            return {}
        # Number of data points determines the next index
        with open(self.data_path) as f:
            data_len = sum(1 for _ in f) - 1  # exclude header
        next_idx = np.array([[data_len]])
        return {
            "cpu_percent": float(self.cpu_model.predict(next_idx)[0]),
            "memory_percent": float(self.mem_model.predict(next_idx)[0]),
        }
