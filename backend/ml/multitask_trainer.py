from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from .feature_extractor import FeatureExtractor


class MultiTaskTrainer:
    """Train separate models for multiple tasks with a shared encoder.

    Each task dataset must be a CSV file containing ``text`` and ``target`` columns.
    The :class:`FeatureExtractor` is fitted on the union of all task texts and
    reused for every individual model.
    """

    def __init__(self, tasks: Dict[str, str]):
        # Map task name to dataset path
        self.tasks = {name: Path(path) for name, path in tasks.items()}
        self.extractor = FeatureExtractor()
        self._datasets: Dict[str, Tuple[list[str], list[float]]] = {}

    def load_datasets(self) -> None:
        """Load datasets and fit the shared feature extractor."""
        texts: list[str] = []
        for name, path in self.tasks.items():
            df = pd.read_csv(path)
            if "text" not in df.columns or "target" not in df.columns:
                raise ValueError(
                    f"Dataset {path} must contain 'text' and 'target' columns"
                )
            task_texts = df["text"].astype(str).tolist()
            targets = df["target"].values
            self._datasets[name] = (task_texts, targets)
            texts.extend(task_texts)
        if texts:
            self.extractor.fit(texts)

    def train(self) -> Dict[str, Tuple[LinearRegression, float]]:
        """Train a model for each task and return metrics."""
        if not self._datasets:
            self.load_datasets()

        results: Dict[str, Tuple[LinearRegression, float]] = {}
        for name, (texts, targets) in self._datasets.items():
            X = self.extractor.transform(texts)
            X_train, X_test, y_train, y_test = train_test_split(
                X, targets, test_size=0.2, random_state=42
            )
            model = LinearRegression()
            model.fit(X_train, y_train)
            mse = mean_squared_error(y_test, model.predict(X_test))
            results[name] = (model, mse)
        return results
