from __future__ import annotations

from pydantic import BaseSettings, Field, SecretStr


class EnvConfig(BaseSettings):
    """Required environment variables for AutoGPT."""

    openai_api_key: SecretStr = Field(..., env="OPENAI_API_KEY")


def validate_env() -> EnvConfig:
    """Validate required environment variables.

    Returns the parsed ``EnvConfig``. Raises ``ValidationError`` if validation
    fails.
    """

    EnvConfig.update_forward_refs(SecretStr=SecretStr)
    return EnvConfig()
