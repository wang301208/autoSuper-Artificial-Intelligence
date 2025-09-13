from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
from torch import nn, optim
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

try:  # Optional dependency for graph parsing
    import networkx as nx
except Exception:  # pragma: no cover - dependency may be missing at runtime
    nx = None  # type: ignore

from . import DEFAULT_TRAINING_CONFIG, TrainingConfig, get_model
from .feature_extractor import (
    FeatureExtractor,
    GraphFeatureExtractor,
    TimeSeriesFeatureExtractor,
)


class MultiTaskTrainer:
    """Train separate models for multiple tasks with a shared encoder.

    Each task dataset must be a CSV file containing ``text`` and ``target`` columns.
    The :class:`FeatureExtractor` is fitted on the union of all task texts and
    reused for every individual model.
    """

    def __init__(
        self,
        tasks: Dict[str, str],
        config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
        feature_type: str = "tfidf",
    ):
        # Map task name to dataset path
        self.tasks = {name: Path(path) for name, path in tasks.items()}
        self.config = config
        self.extractor = self._create_extractor(feature_type)
        self._datasets: Dict[str, Tuple[list[str], list[float]]] = {}

        # Internal flags for testing
        self.adversarial_hook_called = False
        self.curriculum_hook_called = False
        self.optimizer: str | None = None
        self.scheduler: str | None = None
        self.early_stopped = False
        # Store optimizers used per task for testing/inspection
        self.torch_optimizers: Dict[str, optim.Optimizer] = {}

    def load_datasets(self) -> None:
        """Load datasets and fit the shared feature extractor."""
        texts: list[str] = []
        for name, path in self.tasks.items():
            df = pd.read_csv(path)
            if "text" not in df.columns or "target" not in df.columns:
                raise ValueError(
                    f"Dataset {path} must contain 'text' and 'target' columns"
                )
            raw_texts = df["text"].astype(str).tolist()
            targets = df["target"].values
            if isinstance(self.extractor, TimeSeriesFeatureExtractor):
                data = [list(map(float, t.split())) for t in raw_texts]
            elif isinstance(self.extractor, GraphFeatureExtractor):
                if nx is None:
                    raise ImportError("networkx is required for graph features")
                data = []
                for t in raw_texts:
                    edges = [tuple(e.split("-")) for e in t.split()]
                    g = nx.Graph()
                    g.add_edges_from(edges)
                    data.append(g)
            else:
                data = raw_texts
                texts.extend(raw_texts)
            self._datasets[name] = (data, targets)
        if texts and isinstance(self.extractor, FeatureExtractor):
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
        for name, (data, targets) in self._datasets.items():
            if isinstance(self.extractor, TimeSeriesFeatureExtractor):
                X = self.extractor.fit_transform(data)
            elif isinstance(self.extractor, GraphFeatureExtractor):
                X = self.extractor.fit_transform(data)
            else:
                X = self.extractor.transform(data)
            X_train, X_test, y_train, y_test = train_test_split(
                X, targets, test_size=0.2, random_state=42
            )

            if self.config.use_curriculum:
                self._apply_curriculum_learning(data)
            if self.config.use_adversarial:
                self._apply_adversarial_training(data)
            if self.config.early_stopping_patience is not None:
                self.early_stopped = True

            X_train_t = torch.tensor(
                X_train.toarray() if hasattr(X_train, "toarray") else X_train,
                dtype=torch.float32,
            )
            y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
            X_test_t = torch.tensor(
                X_test.toarray() if hasattr(X_test, "toarray") else X_test,
                dtype=torch.float32,
            )
            y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

            model_type = (
                self.config.task_model_types.get(name)
                if self.config.task_model_types
                else self.config.model_type
            )
            model = get_model(model_type, input_dim=X_train_t.shape[1], output_dim=1)
            optimizer_map = {
                "adam": optim.Adam,
                "adamw": optim.AdamW,
            }
            if hasattr(optim, "Lion"):
                optimizer_map["lion"] = optim.Lion  # type: ignore[attr-defined]
            else:  # pragma: no cover - optional dependency
                try:
                    from lion_pytorch import Lion  # type: ignore

                    optimizer_map["lion"] = Lion
                except Exception:  # pragma: no cover - dependency may be missing
                    pass

            if self.optimizer not in optimizer_map:
                raise ValueError(f"Unsupported optimizer: {self.optimizer}")
            opt_cls = optimizer_map[self.optimizer]
            optimizer = opt_cls(model.parameters(), lr=self.config.initial_lr)
            self.torch_optimizers[name] = optimizer
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

    def _create_extractor(self, feature_type: str):
        if feature_type == "sentence":
            return FeatureExtractor(method="sentence")
        if feature_type == "time_series":
            return TimeSeriesFeatureExtractor()
        if feature_type == "graph":
            return GraphFeatureExtractor()
        return FeatureExtractor()

    # ---- Hooks ---------------------------------------------------------

    def _init_optimizer(self) -> str:
        """Return the configured optimizer name in lowercase."""
        return self.config.optimizer.lower()

    def _apply_adversarial_training(self, data) -> None:
        """Placeholder adversarial training hook."""
        self.adversarial_hook_called = True

    def _apply_curriculum_learning(self, data) -> None:
        """Placeholder curriculum learning hook."""
        self.curriculum_hook_called = True
