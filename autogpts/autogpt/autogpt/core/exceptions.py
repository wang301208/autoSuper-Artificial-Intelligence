from __future__ import annotations

from typing import Optional


class AutoGPTError(Exception):
    """Base exception for all AutoGPT related errors."""

    def __init__(self, message: str, *, hint: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
