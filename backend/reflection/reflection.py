"""Simple post-generation reflection module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

EvaluationFn = Callable[[str], str]
RewriteFn = Callable[[str], str]


@dataclass
class ReflectionModule:
    """Evaluate and rewrite an initial response."""

    evaluate: EvaluationFn | None = None
    rewrite: RewriteFn | None = None

    def __post_init__(self) -> None:
        if self.evaluate is None:
            self.evaluate = self._default_evaluate
        if self.rewrite is None:
            self.rewrite = self._default_rewrite

    def _default_evaluate(self, text: str) -> str:
        """Return a naive quality assessment for ``text``."""

        length = len(text.split())
        return f"response_length={length}"

    def _default_rewrite(self, text: str) -> str:
        """Provide a very small revision for ``text``."""

        return text + " [revised]"

    def reflect(self, text: str) -> Tuple[str, str]:
        """Evaluate ``text`` and return (evaluation, revised_text)."""

        evaluation = self.evaluate(text)
        revised = self.rewrite(text)
        return evaluation, revised
