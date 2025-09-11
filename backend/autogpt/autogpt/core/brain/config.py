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
