from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch import nn, optim

from . import DEFAULT_TRAINING_CONFIG, TrainingConfig, get_model
from .feature_extractor import (
    FeatureExtractor,
    GraphFeatureExtractor,
    TimeSeriesFeatureExtractor,
)


DEFAULT_LOG_FILE = Path("data") / "new_logs.csv"


class ContinualTrainer:
    """Incrementally fine-tune models on collected experience.

    The trainer keeps track of how many samples have been processed and triggers
    training whenever a configured threshold of new samples has been reached.
    Checkpoints are written after each training run.
    """

    def __init__(
        self,
        config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
        log_file: Path = DEFAULT_LOG_FILE,
        feature_type: str = "tfidf",
    ) -> None:
        self.config = config
        self.log_file = log_file
        self.trained_rows = 0
        if self.log_file.exists():
            # Account for header row
            with self.log_file.open("r", newline="") as f:
                self.trained_rows = sum(1 for _ in f) - 1
        self.pending_samples = 0
        self.step = 0

        # Internal flags for testing hooks
        self.adversarial_hook_called = False
        self.curriculum_hook_called = False
        self.ewc_hook_called = False
        self.orthogonal_hook_called = False
        self.optimizer: str | None = None
        self.scheduler: str | None = None
        self.early_stopped = False

        # Simple linear model and feature extractor used for fine-tuning
        self.extractor = self._create_extractor(feature_type)
        self.model: nn.Module | None = None
        self.criterion = nn.MSELoss()
        self.torch_optimizer: optim.Optimizer | None = None

        self._load_checkpoint()

    def add_sample(self, sample: Dict[str, Any]) -> None:
        """Register a new sample and trigger training if needed."""
        self.pending_samples += 1
        if self.pending_samples >= self.config.train_after_samples:
            self.train()

    def train(self) -> None:
        """Fine-tune the model on newly collected samples."""
        new_data: List[Dict[str, Any]] = []
        if self.log_file.exists():
            with self.log_file.open("r", newline="") as f:
                reader = list(csv.DictReader(f))
            new_data = reader[self.trained_rows :]
        if not new_data:
            return

        # Select optimizer and scheduler according to config
        self.optimizer = self._init_optimizer()
        self.scheduler = self.config.lr_scheduler

        # Hooks for curriculum learning and adversarial training
        if self.config.use_curriculum:
            self._apply_curriculum_learning(new_data)
        if self.config.use_adversarial:
            self._apply_adversarial_training(new_data)
        if self.config.use_ewc:
            self._apply_ewc_regularization(new_data)
        if self.config.use_orthogonal:
            self._apply_orthogonal_training(new_data)

        if self.config.early_stopping_patience is not None:
            # Placeholder: mark that early stopping would be engaged
            self.early_stopped = True

        rewards = torch.tensor(
            [float(s.get("reward", 0.0)) for s in new_data], dtype=torch.float32
        ).unsqueeze(1)

        if isinstance(self.extractor, TimeSeriesFeatureExtractor):
            series_list = [s.get("series", []) for s in new_data]
            feats = self.extractor.fit_transform(series_list)
            inputs = torch.tensor(feats, dtype=torch.float32)
        elif isinstance(self.extractor, GraphFeatureExtractor):
            graphs = [s.get("graph") for s in new_data]
            feats = self.extractor.fit_transform(graphs)
            inputs = torch.tensor(feats, dtype=torch.float32)
        else:
            texts = []
            for s in new_data:
                joined = " ".join(str(v) for k, v in s.items() if k != "reward")
                if not any(len(tok) > 1 for tok in joined.split()):
                    joined += " filler"
                texts.append(joined)
            try:
                feats = self.extractor.transform(texts)
            except Exception:
                feats = self.extractor.fit_transform(texts)
            arr = feats.toarray() if hasattr(feats, "toarray") else feats
            inputs = torch.tensor(arr, dtype=torch.float32)

        if self.model is None or getattr(self.model, "input_dim", inputs.shape[1]) != inputs.shape[1]:
            self.model = get_model(
                self.config.model_type, input_dim=inputs.shape[1], output_dim=1
            )
            # Map configured optimizer name to implementation class
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
            self.torch_optimizer = opt_cls(
                self.model.parameters(), lr=self.config.initial_lr
            )

        assert self.model is not None
        assert self.torch_optimizer is not None

        self.model.train()
        preds = self.model(inputs)
        loss = self.criterion(preds, rewards)
        self.torch_optimizer.zero_grad()
        loss.backward()
        self.torch_optimizer.step()

        self.trained_rows += len(new_data)
        self.pending_samples = 0
        self.step += 1
        self._save_checkpoint()

    def _save_checkpoint(self) -> None:
        if self.model is None:
            return
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.config.checkpoint_dir / f"checkpoint_{self.step}.pt"
        input_dim = getattr(self.model, "in_features", getattr(self.model, "input_dim", None))
        state = {
            "model_state": self.model.state_dict(),
            "input_dim": input_dim,
            "step": self.step,
            "trained_rows": self.trained_rows,
        }
        if hasattr(self.extractor, "vectorizer"):
            state["vectorizer"] = self.extractor.vectorizer  # type: ignore[attr-defined]
        torch.save(state, ckpt_path)

    def _load_checkpoint(self) -> None:
        if not self.config.checkpoint_dir.exists():
            return
        ckpts = sorted(self.config.checkpoint_dir.glob("checkpoint_*.pt"))
        if not ckpts:
            return
        state = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        input_dim = state.get("input_dim")
        if input_dim is not None:
            self.model = get_model(
                self.config.model_type, input_dim=input_dim, output_dim=1
            )
            self.model.load_state_dict(state["model_state"])
            self.torch_optimizer = optim.Adam(
                self.model.parameters(), lr=self.config.initial_lr
            )
        if "vectorizer" in state and hasattr(self.extractor, "vectorizer"):
            self.extractor.vectorizer = state["vectorizer"]  # type: ignore[attr-defined]
        self.step = state.get("step", 0)
        self.trained_rows = state.get("trained_rows", self.trained_rows)

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

    def _apply_adversarial_training(self, data: List[Dict[str, Any]]) -> None:
        """Placeholder adversarial training hook."""
        self.adversarial_hook_called = True

    def _apply_curriculum_learning(self, data: List[Dict[str, Any]]) -> None:
        """Placeholder curriculum learning hook."""
        self.curriculum_hook_called = True

    def _apply_ewc_regularization(self, data: List[Dict[str, Any]]) -> None:
        """Placeholder Elastic Weight Consolidation regularization."""
        self.ewc_hook_called = True

    def _apply_orthogonal_training(self, data: List[Dict[str, Any]]) -> None:
        """Placeholder orthogonal gradient descent step."""
        self.orthogonal_hook_called = True
