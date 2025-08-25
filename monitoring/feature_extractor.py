"""Feature extraction utilities for converting raw log lines into numerical vectors."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureExtractor:
    """Convert collections of log strings into numerical feature vectors.

    The extractor internally uses :class:`~sklearn.feature_extraction.text.TfidfVectorizer`
    to transform raw log text into a sparse matrix of TF-IDF features.
    """

    def __init__(self, **vectorizer_kwargs) -> None:
        self.vectorizer = TfidfVectorizer(**vectorizer_kwargs)

    # ------------------------------------------------------------------
    # fitting and transforming
    # ------------------------------------------------------------------
    def fit(self, logs: Iterable[str]) -> "FeatureExtractor":
        """Learn the vocabulary and IDF weights from *logs*."""
        self.vectorizer.fit(list(logs))
        return self

    def transform(self, logs: Iterable[str]):
        """Transform *logs* into a sparse feature matrix."""
        return self.vectorizer.transform(list(logs))

    def fit_transform(self, logs: Iterable[str]):
        """Fit to *logs* then return the transformed matrix."""
        return self.vectorizer.fit_transform(list(logs))

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, path: Path | str) -> None:
        """Serialize the underlying vectorizer to *path*."""
        joblib.dump(self.vectorizer, Path(path))

    @classmethod
    def load(cls, path: Path | str) -> "FeatureExtractor":
        """Load a previously saved extractor from *path*."""
        inst = cls()
        inst.vectorizer = joblib.load(Path(path))
        return inst
