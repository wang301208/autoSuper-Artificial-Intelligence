"""Simple meta-learning components for few-shot adaptation.

This module implements small, self-contained utilities that mimic the
behaviour of meta-learning algorithms.  The goal is not to be
state-of-the-art but to provide an easy to understand interface that can
be used in tests and examples.  All implementations operate on tiny
linear models represented by NumPy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np


@dataclass
class FewShotTask:
    """Container holding a single few-shot task.

    Parameters
    ----------
    support_x, support_y:
        Arrays forming the support set used for inner adaptation.
    query_x, query_y:
        Arrays forming the query set used for meta-updates or
        evaluation after adaptation.
    """

    support_x: np.ndarray
    support_y: np.ndarray
    query_x: np.ndarray
    query_y: np.ndarray


class MetaMemorySystem:
    """Very small memory storing performance of previous tasks."""

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def store(self, info: Dict[str, Any]) -> None:
        """Store a record describing a training step."""

        self.records.append(dict(info))

    def recall_best(self) -> Dict[str, Any] | None:
        """Return the record with the lowest loss if available."""

        if not self.records:
            return None
        return min(self.records, key=lambda r: r.get("loss", float("inf")))


class SelfReflectionModule:
    """Produce textual reflections based on improvement in loss."""

    def __init__(self) -> None:
        self.history: List[str] = []

    def assess(self, loss_before: float, loss_after: float) -> str:
        """Generate a simple reflection string.

        The message mentions how much the loss improved (or worsened).
        """

        delta = loss_before - loss_after
        msg = f"loss changed by {delta:.4f}"
        self.history.append(msg)
        return msg


class ReptileOptimizer:
    """Minimal Reptile style meta-optimizer.

    The optimiser performs inner-loop adaptation for each task and moves
    the meta-parameters towards the adapted parameters.
    """

    def __init__(self, inner_lr: float = 0.1, meta_lr: float = 0.1, adapt_steps: int = 1) -> None:
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.adapt_steps = adapt_steps

    def _loss_and_grad(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        preds = X @ w
        diff = preds - y
        loss = float(np.mean(diff ** 2))
        grad = 2 * X.T @ diff / len(X)
        return loss, grad

    def adapt(self, w: np.ndarray, task: FewShotTask) -> np.ndarray:
        """Return task adapted weights using ``inner_lr`` gradient steps."""

        w = w.copy()
        for _ in range(self.adapt_steps):
            _, grad = self._loss_and_grad(w, task.support_x, task.support_y)
            w -= self.inner_lr * grad
        return w

    def meta_update(self, w: np.ndarray, tasks: List[FewShotTask]) -> np.ndarray:
        """Apply a single Reptile meta-update across ``tasks``."""

        w = w.copy()
        for task in tasks:
            adapted = self.adapt(w, task)
            w += self.meta_lr * (adapted - w)
        return w


class MAMLEngine:
    """Simplified MAML style learner with memory and reflection."""

    def __init__(
        self,
        input_dim: int,
        inner_lr: float = 0.1,
        meta_lr: float = 0.1,
        adapt_steps: int = 1,
        algorithm: str = "maml",
    ) -> None:
        self.weights = np.zeros(input_dim)
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.adapt_steps = adapt_steps
        self.algorithm = algorithm
        self.memory = MetaMemorySystem()
        self.reflection = SelfReflectionModule()
        self.reptile = ReptileOptimizer(inner_lr, meta_lr, adapt_steps)

    def _loss_and_grad(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        preds = X @ w
        diff = preds - y
        loss = float(np.mean(diff ** 2))
        grad = 2 * X.T @ diff / len(X)
        return loss, grad

    def _adapt(self, w: np.ndarray, task: FewShotTask, steps: int | None = None) -> np.ndarray:
        """Inner-loop adaptation on the support set."""

        steps = steps or self.adapt_steps
        w = w.copy()
        for _ in range(steps):
            _, grad = self._loss_and_grad(w, task.support_x, task.support_y)
            w -= self.inner_lr * grad
        return w

    def learn_to_learn(self, tasks: List[FewShotTask], epochs: int = 1) -> List[float]:
        """Meta-train across ``tasks`` for ``epochs`` iterations."""

        history: List[float] = []
        for _ in range(epochs):
            if self.algorithm == "reptile":
                self.weights = self.reptile.meta_update(self.weights, tasks)
                # Evaluate average loss for logging.
                losses = [self._loss_and_grad(self.weights, t.query_x, t.query_y)[0] for t in tasks]
                avg_loss = float(np.mean(losses))
                history.append(avg_loss)
                self.memory.store({"loss": avg_loss, "weights": self.weights.copy()})
                continue

            meta_grad = np.zeros_like(self.weights)
            epoch_loss = 0.0
            for task in tasks:
                adapted = self._adapt(self.weights, task)
                loss, grad = self._loss_and_grad(adapted, task.query_x, task.query_y)
                epoch_loss += loss
                meta_grad += grad
            self.weights -= self.meta_lr * meta_grad / len(tasks)
            avg_loss = epoch_loss / len(tasks)
            history.append(avg_loss)
            self.memory.store({"loss": avg_loss, "weights": self.weights.copy()})
        return history

    def fast_adapt_to_task(self, task: FewShotTask, steps: int | None = None) -> tuple[np.ndarray, str]:
        """Quickly adapt to ``task`` using current meta-parameters.

        Returns the adapted weights and a textual reflection from the
        :class:`SelfReflectionModule`.
        """

        before_loss, _ = self._loss_and_grad(self.weights, task.query_x, task.query_y)
        adapted = self._adapt(self.weights, task, steps)
        after_loss, _ = self._loss_and_grad(adapted, task.query_x, task.query_y)
        self.memory.store({"loss": after_loss, "weights": adapted.copy()})
        reflection = self.reflection.assess(before_loss, after_loss)
        return adapted, reflection


__all__ = [
    "FewShotTask",
    "MetaMemorySystem",
    "SelfReflectionModule",
    "ReptileOptimizer",
    "MAMLEngine",
]
