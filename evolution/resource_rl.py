"""Reinforcement learning agent for resource allocation using DQN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import torch
from torch import nn, optim


class _DQN(nn.Module):
    def __init__(self, state_dim: int = 2, action_dim: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float


class ResourceRL:
    """Minimal DQN agent for resource optimisation."""

    def __init__(self, gamma: float = 0.9) -> None:
        self.gamma = gamma
        self.policy = _DQN()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.01)

    # ------------------------------------------------------------------
    def select_action(self, cpu: float, mem: float) -> int:
        """Choose an action based on current predictions."""

        state = torch.tensor([[cpu, mem]], dtype=torch.float32)
        with torch.no_grad():
            q_vals = self.policy(state)
        return int(torch.argmax(q_vals).item())

    # ------------------------------------------------------------------
    def train(self, transitions: Iterable[Transition], epochs: int = 50) -> None:
        """Train the DQN from a batch of transitions."""

        states: List[np.ndarray] = []
        targets: List[float] = []
        actions: List[int] = []
        for t in transitions:
            states.append(t.state)
            targets.append(t.reward)
            actions.append(t.action)

        if not states:
            return
        x = torch.tensor(states, dtype=torch.float32)
        y = torch.tensor(targets, dtype=torch.float32)
        a = torch.tensor(actions, dtype=torch.long)
        for _ in range(epochs):
            q_vals = self.policy(x).gather(1, a.unsqueeze(1)).squeeze()
            loss = nn.functional.mse_loss(q_vals, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # ------------------------------------------------------------------
    def evaluate(self, samples: Iterable[Tuple[float, float]]) -> float:
        """Return average reward for given samples."""

        total = 0.0
        n = 0
        for cpu, mem in samples:
            state = np.array([cpu, mem])
            action = self.select_action(cpu, mem)
            reward = self._rule_reward(state, action)
            total += reward
            n += 1
        return total / max(n, 1)

    # ------------------------------------------------------------------
    @staticmethod
    def _rule_reward(state: np.ndarray, action: int) -> float:
        """Simple rule-based reward used for training/evaluation.

        Action mapping: 0=scale down, 1=keep, 2=scale up.
        Reward is +1 for correct decision according to thresholds, else -1.
        """

        cpu, mem = state
        if cpu > 80 or mem > 80:
            return 1.0 if action == 0 else -1.0
        if cpu < 20 and mem < 20:
            return 1.0 if action == 2 else -1.0
        return 1.0 if action == 1 else -1.0


if __name__ == "__main__":  # pragma: no cover - manual demo
    # Simple training and evaluation script
    data = [np.array([10.0, 10.0]), np.array([90.0, 90.0]), np.array([50.0, 50.0])]
    agent = ResourceRL()
    transitions = [
        Transition(state, action=0, reward=agent._rule_reward(state, 0)) for state in data
    ]
    agent.train(transitions)
    print("Average reward", agent.evaluate((s for s in data)))

