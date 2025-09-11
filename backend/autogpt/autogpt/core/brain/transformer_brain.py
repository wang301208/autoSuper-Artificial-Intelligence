from __future__ import annotations

import torch
from torch import nn

from .config import TransformerBrainConfig


class TransformerBrain(nn.Module):
    """Simple transformer based "brain" used for internal planning."""

    def __init__(self, config: TransformerBrainConfig | None = None):
        super().__init__()
        self.config = config or TransformerBrainConfig()

        self.embedding = nn.Linear(self.config.dim, self.config.dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.dim,
            nhead=self.config.heads,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.config.layers)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.config.dim,
            num_heads=self.config.heads,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.action_head = nn.Linear(self.config.dim, self.config.dim)

    def think(self, observation, memory_ctx=None):
        """Encode observation and optional memory context into a thought vector."""

        obs = torch.as_tensor(observation, dtype=torch.float32)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        x = self.embedding(obs)
        x = self.transformer(x)

        if memory_ctx is not None:
            mem = torch.as_tensor(memory_ctx, dtype=torch.float32)
            if mem.dim() == 1:
                mem = mem.unsqueeze(0)
            mem = self.embedding(mem)
            x, _ = self.cross_attn(x, mem, mem)

        thought = x.mean(dim=0).squeeze(0)
        return thought

    def propose_action(self, thought):
        """Generate an action proposal based on the thought vector."""

        t = torch.as_tensor(thought, dtype=torch.float32)
        action_vec = self.action_head(t)
        return (
            "internal_brain_action",
            {},
            {
                "thought": t.detach().cpu().tolist(),
                "action": action_vec.detach().cpu().tolist(),
            },
        )
