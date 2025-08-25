"""Utilities for converting raw log lines into numerical feature vectors."""
from __future__ import annotations

from typing import Iterable, List

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


class FeatureExtractor:
    """Converts raw log strings into TF-IDF feature vectors.

    The extractor wraps a ``TfidfVectorizer`` so that we can easily serialise the
    fitted vocabulary and reuse it for future data.
    """

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer()

    def fit_transform(self, logs: Iterable[str]):
        """Fit the vectoriser on ``logs`` and return the transformed matrix."""
        return self.vectorizer.fit_transform(list(logs))

    def transform(self, logs: Iterable[str]):
        """Transform ``logs`` using the fitted vectoriser."""
        return self.vectorizer.transform(list(logs))

    def save(self, path: str) -> None:
        """Serialise the underlying vectoriser to ``path`` using ``joblib``."""
        joblib.dump(self.vectorizer, path)

    @classmethod
    def load(cls, path: str) -> "FeatureExtractor":
        """Load a previously saved extractor from ``path``."""
        instance = cls()
        instance.vectorizer = joblib.load(path)
        return instance
