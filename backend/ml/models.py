"""Model architectures and factory helpers for AutoGPT training."""
from __future__ import annotations

from typing import Dict, Type


try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
except Exception:  # pragma: no cover - torch may be missing at runtime
    torch = None  # type: ignore

    class _StubModule:  # minimal placeholder to allow subclassing
        pass

    class nn:  # type: ignore
        Module = _StubModule

try:  # pragma: no cover - optional dependency
    from transformers import AutoModel
except Exception:  # pragma: no cover - transformers may be missing
    AutoModel = None  # type: ignore


class TransformerTextModel(nn.Module):
    """Wrapper around ``transformers.AutoModel`` for text tasks."""

    def __init__(self, model_name: str = "distilbert-base-uncased") -> None:
        if AutoModel is None:  # pragma: no cover - runtime dependency check
            raise ImportError(
                "transformers package is required for TransformerTextModel"
            )
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, *args, **kwargs):  # type: ignore[override]
        return self.model(*args, **kwargs)


class VisionCNN(nn.Module):
    """Simple convolutional network for vision tasks."""

    def __init__(self, num_classes: int = 10) -> None:
        if torch is None:  # pragma: no cover - runtime dependency check
            raise ImportError("torch is required for VisionCNN")
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(64 * 8 * 8, num_classes)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[override]
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class SequenceRNN(nn.Module):
    """Recurrent network (LSTM/GRU) for sequential data."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        rnn_type: str = "lstm",
    ) -> None:
        if torch is None:  # pragma: no cover - runtime dependency check
            raise ImportError("torch is required for SequenceRNN")
        super().__init__()
        rnn_cls = nn.LSTM if rnn_type.lower() == "lstm" else nn.GRU
        self.rnn = rnn_cls(input_size, hidden_size, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[override]
        out, _ = self.rnn(x)
        return self.output(out[:, -1, :])


_MODELS: Dict[str, Type[nn.Module]] = {
    "transformer": TransformerTextModel,
    "vision_cnn": VisionCNN,
    "sequence_rnn": SequenceRNN,
}


def get_model(model_type: str, **kwargs) -> nn.Module:
    """Instantiate a model by type name.

    Parameters
    ----------
    model_type:
        Key identifying the model class. Supported values are
        ``"transformer"``, ``"vision_cnn"`` and ``"sequence_rnn"``.
    **kwargs:
        Additional arguments passed to the model constructor.
    """

    model_cls = _MODELS.get(model_type.lower())
    if model_cls is None:
        raise ValueError(f"Unknown model type: {model_type}")
    return model_cls(**kwargs)


__all__ = [
    "TransformerTextModel",
    "VisionCNN",
    "SequenceRNN",
    "get_model",
]
