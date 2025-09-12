from __future__ import annotations

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from .transformer_brain import TransformerBrain
from .config import TransformerBrainConfig


class ObservationActionDataset(Dataset):
    """Simple dataset mapping observations to action indices."""

    def __init__(self, size: int, dim: int):
        self.size = size
        self.dim = dim
        self.observations = torch.randn(size, dim)
        self.actions = torch.randint(0, dim, (size,))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        return self.observations[index], self.actions[index]


def save_brain(brain: TransformerBrain, path: str) -> None:
    """Persist model weights to ``path``."""

    torch.save(brain.state_dict(), path)


def load_brain(config: TransformerBrainConfig, path: str) -> TransformerBrain:
    """Load model weights from ``path`` into a new ``TransformerBrain`` instance."""

    brain = TransformerBrain(config)
    brain.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    return brain


def train(
    brain: TransformerBrain,
    dataset: Dataset,
    *,
    learning_rate: float | None = None,
    epochs: int | None = None,
) -> TransformerBrain:
    """Train ``brain`` on ``dataset`` using cross-entropy loss."""

    config = brain.config
    lr = learning_rate or config.learning_rate
    num_epochs = epochs or config.epochs
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(brain.parameters(), lr=lr)

    brain.train()
    for _ in range(num_epochs):
        for observations, target in dataloader:
            optimizer.zero_grad()
            logits = []
            for obs in observations:
                thought = brain.think(obs)
                logits.append(brain.action_head(thought))
            logits = torch.stack(logits)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
    return brain


def main():  # pragma: no cover - example usage
    config = TransformerBrainConfig()
    brain = TransformerBrain(config)
    dataset = ObservationActionDataset(256, config.dim)
    train(brain, dataset)
    save_brain(brain, "transformer_brain.pth")


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
