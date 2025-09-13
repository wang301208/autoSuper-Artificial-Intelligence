"""Example algorithms for demonstrating the A/B testing utilities."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from typing import Any, Sequence


def algo_random(X: Any, y: Sequence[int]) -> Sequence[int]:
    """Predict labels uniformly at random."""

    model = DummyClassifier(strategy="uniform", random_state=0)
    model.fit(X, y)
    return model.predict(X)


def algo_knn(X: Any, y: Sequence[int]) -> Sequence[int]:
    """Simple k-nearest neighbours classifier."""

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    return model.predict(X)
