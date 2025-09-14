from __future__ import annotations

"""Adaptive fitness generator for multi-objective evolutionary runs.

The :class:`AdaptiveFitnessGenerator` dynamically adjusts weights assigned to
multiple objective functions based on historical performance and optional
external environment signals.  The generator can be used as a drop-in fitness
function for the :class:`~modules.evolution.generic_ga.GeneticAlgorithm` by
passing :meth:`evaluate` as the ``fitness_fn`` argument.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence


@dataclass
class AdaptiveFitnessGenerator:
    """Compose a fitness function with dynamically learned weights.

    Parameters
    ----------
    objectives
        Mapping from objective name to a callable returning a fitness score for
        a given individual.  Higher scores are assumed to be better.
    initial_weights
        Optional mapping of initial weights for each objective.  When omitted,
        all objectives are weighted uniformly.
    history_length
        Number of recent evaluations considered when estimating performance
        trends.
    learning_rate
        Step size used when adjusting weights.
    """

    objectives: Dict[str, Callable[[Sequence[float]], float]]
    initial_weights: Dict[str, float] | None = None
    history_length: int = 50
    learning_rate: float = 0.1
    _history: List[Dict[str, float]] = field(default_factory=list, init=False)
    _env_signal: Dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.initial_weights is None:
            weight = 1.0 / len(self.objectives)
            self.weights: Dict[str, float] = {name: weight for name in self.objectives}
        else:
            self.weights = dict(self.initial_weights)
            self._normalise_weights()

    # ------------------------------------------------------------------
    # Public API
    def update_environment(self, signals: Dict[str, float]) -> None:
        """Update external environment signals.

        Signals are expected to be in the range ``[-1, 1]`` where positive
        values increase the importance of the corresponding objective and
        negative values decrease it.
        """

        self._env_signal.update(signals)

    def evaluate(self, individual: Sequence[float]) -> float:
        """Evaluate an individual and update objective weights."""

        scores = {name: fn(individual) for name, fn in self.objectives.items()}
        self._history.append(scores)
        self._adjust_weights(scores)
        return sum(self.weights[name] * score for name, score in scores.items())

    # ------------------------------------------------------------------
    # Weight adaptation utilities
    def _adjust_weights(self, scores: Dict[str, float]) -> None:
        """Adjust weights based on score trends and environment signals."""

        window = self._history[-self.history_length :]
        if len(window) > 1:
            avgs = {
                name: sum(h[name] for h in window) / len(window)
                for name in scores
            }
            for name, score in scores.items():
                trend = score - avgs[name]
                # When an objective's performance drops below its moving
                # average, increase its weight to emphasise improvement.  Clamp
                # to keep weights non-negative.
                self.weights[name] = max(
                    0.0, self.weights[name] - self.learning_rate * trend
                )

        for name, signal in self._env_signal.items():
            if name in self.weights:
                self.weights[name] *= 1 + self.learning_rate * signal

        self._normalise_weights()

    def _normalise_weights(self) -> None:
        total = sum(self.weights.values())
        if total <= 0:
            # Avoid division by zero and keep equal weights as fallback
            weight = 1.0 / len(self.weights)
            for name in self.weights:
                self.weights[name] = weight
            return
        for name in self.weights:
            self.weights[name] /= total
