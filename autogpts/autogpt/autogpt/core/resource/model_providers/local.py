
"""Simple local embedding provider used for tests and offline operation."""
from __future__ import annotations

import numpy as np

from .schema import (
    EmbeddingModelProvider,
    EmbeddingModelResponse,
    EmbeddingModelInfo,
    ModelProviderName,
    ModelProviderSettings,
    ModelProviderConfiguration,
    ModelProviderCredentials,
    ModelTokenizer,
)


class _SimpleTokenizer:
    def encode(self, text: str) -> list[str]:
        return text.split()

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)


class LocalEmbeddingProvider(EmbeddingModelProvider):
    """Deterministic embedding provider using hashed bag-of-words vectors."""

    default_settings = ModelProviderSettings(
        configuration=ModelProviderConfiguration(),
        credentials=ModelProviderCredentials(),
    )

    def __init__(self, settings: ModelProviderSettings | None = None) -> None:
        if settings is None:
            settings = self.default_settings
        self._configuration = settings.configuration
        self._budget = settings.budget

    # Tokenizer utilities -------------------------------------------------
    def count_tokens(self, text: str, model_name: str) -> int:  # pragma: no cover - trivial
        return len(self.get_tokenizer(model_name).encode(text))

    def get_tokenizer(self, model_name: str) -> ModelTokenizer:  # pragma: no cover - trivial
        return _SimpleTokenizer()

    def get_token_limit(self, model_name: str) -> int:  # pragma: no cover - simple constant
        return 8192

    # Embedding generation ------------------------------------------------
    async def create_embedding(
        self,
        text: str,
        model_name: str,
        embedding_parser,
        **kwargs,
    ) -> EmbeddingModelResponse:
        tokens = self.get_tokenizer(model_name).encode(text.lower())
        dims = 64
        vec = np.zeros(dims, dtype=np.float32)
        for tok in tokens:
            vec[hash(tok) % dims] += 1.0
        info = EmbeddingModelInfo(
            name=model_name,
            provider_name=ModelProviderName.LOCAL,
            prompt_token_cost=0.0,
            completion_token_cost=0.0,
            max_tokens=dims,
            embedding_dimensions=dims,
        )
        return EmbeddingModelResponse(
            embedding=embedding_parser(vec.tolist()),
            prompt_tokens_used=len(tokens),
            completion_tokens_used=0,
            model_info=info,
        )
