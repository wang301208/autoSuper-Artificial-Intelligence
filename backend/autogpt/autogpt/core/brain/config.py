from __future__ import annotations

from autogpt.core.configuration import SystemConfiguration, UserConfigurable


class TransformerBrainConfig(SystemConfiguration):
    """Configuration options for the internal transformer brain."""

    layers: int = UserConfigurable(default=2)
    """Number of transformer encoder layers."""

    heads: int = UserConfigurable(default=4)
    """Number of attention heads in each transformer layer."""

    dim: int = UserConfigurable(default=256)
    """Dimensionality of the model."""

    dropout: float = UserConfigurable(default=0.1)
    """Dropout probability used in transformer layers."""

    # Training configuration
    learning_rate: float = UserConfigurable(default=1e-3)
    """Learning rate used during training."""

    epochs: int = UserConfigurable(default=10)
    """Number of training epochs."""

    batch_size: int = UserConfigurable(default=4)
    """Batch size for the training dataloader."""
