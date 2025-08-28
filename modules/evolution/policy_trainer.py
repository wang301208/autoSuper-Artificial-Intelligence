from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
import math


@dataclass
class PolicyTrainer:
    """Simple REINFORCE-style trainer for discrete policies.

    Experiences are tuples of (state, action, reward). The trainer can read
    additional experiences from a CSV dataset and update a policy mapping
    states to action-preference values.
    """

    dataset_path: Path
    learning_rate: float = 0.1
    policy: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dataset_path = Path(self.dataset_path)
        self._buffer: List[Tuple[str, str, float]] = []

    # ------------------------------------------------------------------
    # Experience management
    # ------------------------------------------------------------------
    def push_experience(self, state: str, action: str, reward: float) -> None:
        """Add an experience tuple to the in-memory buffer."""
        self._buffer.append((state, action, reward))

    def _load_dataset(self) -> List[Tuple[str, str, float]]:
        experiences: List[Tuple[str, str, float]] = []
        if not self.dataset_path.exists():
            return experiences
        try:
            with self.dataset_path.open(newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        experiences.append(
                            (row["state"], row["action"], float(row["reward"]))
                        )
                    except (KeyError, ValueError):
                        continue
        except Exception:
            pass
        return experiences

    # ------------------------------------------------------------------
    # Policy update
    # ------------------------------------------------------------------
    def _softmax(self, values: List[float]) -> List[float]:
        if not values:
            return []
        max_v = max(values)
        exps = [math.exp(v - max_v) for v in values]
        total = sum(exps) or 1.0
        return [e / total for e in exps]

    def update_policy(self) -> Dict[str, Dict[str, float]]:
        """Update the policy using buffered and dataset experiences.

        Returns the updated policy mapping.
        """
        experiences = self._load_dataset() + self._buffer
        for state, action, reward in experiences:
            prefs = self.policy.setdefault(state, {})
            prefs.setdefault(action, 0.0)
            actions = list(prefs.keys())
            probs = self._softmax([prefs[a] for a in actions])
            for name, prob in zip(actions, probs):
                indicator = 1.0 if name == action else 0.0
                prefs[name] += self.learning_rate * reward * (indicator - prob)
        self._buffer.clear()
        return self.policy
