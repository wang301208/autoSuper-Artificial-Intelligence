"""Deep learning model for predicting system resource usage with RL strategy."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn, optim

from .resource_rl import ResourceRL

logger = logging.getLogger(__name__)


class _ResourceNet(nn.Module):
    """Simple two-headed network predicting CPU and memory usage."""

    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU())
        self.cpu_head = nn.Linear(16, 1)
        self.mem_head = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        x = self.shared(x)
        return self.cpu_head(x), self.mem_head(x)


class ResourceModel:
    """Train models on historical metrics and predict future usage."""

    def __init__(
        self,
        data_path: Path | str | None = None,
        rl_agent: Optional[ResourceRL] = None,
    ) -> None:
        self.data_path = (
            Path(data_path) if data_path is not None else Path(__file__).with_name("metrics_history.csv")
        )
        self.model_path = self.data_path.with_suffix(".pt")
        self.net = _ResourceNet()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.01)
        self.rl_agent = rl_agent or ResourceRL()
        self._trained = False

    # ------------------------------------------------------------------
    def _load(self) -> np.ndarray:
        data: list[tuple[float, float]] = []
        try:
            with open(self.data_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append((float(row["cpu_percent"]), float(row["memory_percent"])))
        except FileNotFoundError:
            return np.empty((0, 2))
        return np.array(data)

    # ------------------------------------------------------------------
    def train(self, epochs: int = 100) -> None:
        """Train the neural network on historical data."""

        data = self._load()
        if data.size == 0:
            logger.warning("No historical data found for training")
            self._trained = False
            return
        x = torch.arange(len(data), dtype=torch.float32).unsqueeze(1)
        y_cpu = torch.tensor(data[:, 0], dtype=torch.float32).unsqueeze(1)
        y_mem = torch.tensor(data[:, 1], dtype=torch.float32).unsqueeze(1)
        self.net.train()
        for _ in range(epochs):
            cpu_pred, mem_pred = self.net(x)
            loss = self.criterion(cpu_pred, y_cpu) + self.criterion(mem_pred, y_mem)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        torch.save(self.net.state_dict(), self.model_path)
        logger.info("Resource model trained; final loss %.4f", float(loss))
        self._trained = True

    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load model weights if available."""

        if self.model_path.exists():
            self.net.load_state_dict(torch.load(self.model_path, map_location="cpu"))
            self._trained = True

    # ------------------------------------------------------------------
    def predict_next(self) -> Dict[str, float]:
        """Predict next CPU and memory usage values and propose action."""

        if not self._trained:
            self.load()
        if not self._trained:
            self.train()
        if not self._trained:
            return {}

        with open(self.data_path) as f:
            data_len = sum(1 for _ in f) - 1  # exclude header
        x = torch.tensor([[float(data_len)]])
        self.net.eval()
        with torch.no_grad():
            cpu_pred, mem_pred = self.net(x)
        cpu_val = float(cpu_pred.item())
        mem_val = float(mem_pred.item())
        action = self.rl_agent.select_action(cpu_val, mem_val)
        logger.info("Predicted CPU %.2f, MEM %.2f, action %s", cpu_val, mem_val, action)
        return {"cpu_percent": cpu_val, "memory_percent": mem_val, "action": action}

