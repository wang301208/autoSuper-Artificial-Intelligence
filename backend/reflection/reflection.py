"""Post-generation reflection backed by an LLM."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Tuple

EvaluationFn = Callable[[str], str]
RewriteFn = Callable[[str], str]

try:  # pragma: no cover - optional dependency
    import openai  # type: ignore
except Exception:  # pragma: no cover - handled gracefully
    openai = None

logger = logging.getLogger(__name__)


@dataclass
class ReflectionModule:
    """Evaluate and rewrite an initial response using an LLM.

    Parameters
    ----------
    evaluate
        Optional custom evaluation function. If not provided a call to
        :func:`openai.ChatCompletion.create` is attempted and the result is
        interpreted as a ``score`` between 0 and 1.
    rewrite
        Optional custom rewrite function. If not provided the LLM is asked to
        improve the text while keeping the meaning.
    max_passes
        Maximum number of reflection passes to attempt.
    quality_threshold
        Minimum acceptable score before stopping further reflection passes.
    model
        Name of the LLM model to use when calling the OpenAI API.
    history
        List of tuples containing ``(evaluation, revised_text)`` for each pass.
        This acts as a log for later analysis or fine-tuning.
    """

    evaluate: EvaluationFn | None = None
    rewrite: RewriteFn | None = None
    max_passes: int = 1
    quality_threshold: float = 0.0
    model: str = "gpt-3.5-turbo"
    history: list[Tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.evaluate is None:
            self.evaluate = self._llm_evaluate
        if self.rewrite is None:
            self.rewrite = self._llm_rewrite

    # --- Default LLM-backed implementations ---------------------------------

    def _llm_evaluate(self, text: str) -> str:
        """Return a quality assessment for ``text`` using an LLM.

        Falls back to a trivial heuristic if the API is unavailable.
        """

        if openai is not None and os.getenv("OPENAI_API_KEY"):
            try:  # pragma: no cover - network call
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "Score the quality of the following response "
                                "between 0 and 1 and explain briefly:\n" + text
                            ),
                        }
                    ],
                )
                return response["choices"][0]["message"]["content"].strip()
            except Exception as exc:  # pragma: no cover - network failure
                logger.warning("LLM evaluation failed: %s", exc)
        # Fallback heuristic
        length = len(text.split())
        return f"score=0.0 reason=unavailable length={length}"

    def _llm_rewrite(self, text: str) -> str:
        """Improve ``text`` using an LLM.

        Falls back to appending a tag if the API is unavailable.
        """

        if openai is not None and os.getenv("OPENAI_API_KEY"):
            try:  # pragma: no cover - network call
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "Rewrite the following response to improve "
                                "clarity and quality while preserving meaning:\n"
                                + text
                            ),
                        }
                    ],
                )
                return response["choices"][0]["message"]["content"].strip()
            except Exception as exc:  # pragma: no cover - network failure
                logger.warning("LLM rewrite failed: %s", exc)
        return text + " [revised]"

    # ------------------------------------------------------------------------

    def reflect(self, text: str) -> Tuple[str, str]:
        """Evaluate ``text`` and return ``(evaluation, revised_text)``.

        The method performs up to ``max_passes`` reflection cycles. After each
        evaluation, the score is checked against ``quality_threshold``. If the
        threshold is not met, a rewrite is attempted. Each pass is stored in
        :attr:`history` and logged for later analysis.
        """

        self.history.clear()
        revised = text

        for i in range(self.max_passes):
            evaluation = self.evaluate(revised)
            logger.info("reflection_evaluation_pass_%d: %s", i, evaluation)
            score = self._parse_score(evaluation)
            if score >= self.quality_threshold or i == self.max_passes - 1:
                self.history.append((evaluation, revised))
                return evaluation, revised
            revised = self.rewrite(revised)
            logger.info("reflection_revision_pass_%d: %s", i, revised)
            self.history.append((evaluation, revised))

        return evaluation, revised  # pragma: no cover - loop always returns

    def _parse_score(self, evaluation: str) -> float:
        """Extract a numeric score from an evaluation string."""

        match = re.search(r"([0-9]*\.?[0-9]+)", evaluation)
        try:
            return float(match.group(1)) if match else 0.0
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return 0.0
