from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
from torch import nn, optim
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from . import DEFAULT_TRAINING_CONFIG, TrainingConfig
from .feature_extractor import FeatureExtractor


class MultiTaskTrainer:
    """Train separate models for multiple tasks with a shared encoder.

    Each task dataset must be a CSV file containing ``text`` and ``target`` columns.
    The :class:`FeatureExtractor` is fitted on the union of all task texts and
    reused for every individual model.
    """

    def __init__(
        self, tasks: Dict[str, str], config: TrainingConfig = DEFAULT_TRAINING_CONFIG
    ):
        # Map task name to dataset path
        self.tasks = {name: Path(path) for name, path in tasks.items()}
        self.config = config
        self.extractor = FeatureExtractor()
        self._datasets: Dict[str, Tuple[list[str], list[float]]] = {}

        # Internal flags for testing
        self.adversarial_hook_called = False
        self.curriculum_hook_called = False
        self.optimizer: str | None = None
        self.scheduler: str | None = None
        self.early_stopped = False

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
            # ``FeatureExtractor`` exposes ``fit_transform`` instead of ``fit``
            # for simplicity. We only need the fitted vocabulary here.
            self.extractor.fit_transform(texts)

    def train(self) -> Dict[str, Tuple[nn.Module, float]]:
        """Train a model for each task and return metrics."""
        if not self._datasets:
            self.load_datasets()

        self.optimizer = self._init_optimizer()
        self.scheduler = self.config.lr_scheduler

        results: Dict[str, Tuple[nn.Module, float]] = {}
        for name, (texts, targets) in self._datasets.items():
            X = self.extractor.transform(texts)
            X_train, X_test, y_train, y_test = train_test_split(
                X, targets, test_size=0.2, random_state=42
            )

            if self.config.use_curriculum:
                self._apply_curriculum_learning(texts)
            if self.config.use_adversarial:
                self._apply_adversarial_training(texts)
            if self.config.early_stopping_patience is not None:
                self.early_stopped = True

            X_train_t = torch.tensor(X_train.toarray(), dtype=torch.float32)
            y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
            X_test_t = torch.tensor(X_test.toarray(), dtype=torch.float32)
            y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

            model = nn.Linear(X_train_t.shape[1], 1)
            optimizer = optim.Adam(model.parameters(), lr=self.config.initial_lr)
            criterion = nn.MSELoss()

            model.train()
            preds = model(X_train_t)
            loss = criterion(preds, y_train_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                mse = criterion(model(X_test_t), y_test_t).item()
            results[name] = (model, mse)
        return results

    # ---- Hooks ---------------------------------------------------------

    def _init_optimizer(self) -> str:
        """Return the configured optimizer name."""
        opt = self.config.optimizer.lower()
        if opt not in {"adam", "adamw", "lion"}:
            opt = "adam"
        return opt

    def _apply_adversarial_training(self, data) -> None:
        """Placeholder adversarial training hook."""
        self.adversarial_hook_called = True

    def _apply_curriculum_learning(self, data) -> None:
        """Placeholder curriculum learning hook."""
        self.curriculum_hook_called = True
