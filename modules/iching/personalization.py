"""Personalization utilities for I Ching interpretations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class UserProfile:
    """Stores a user's interaction history and preferences."""

    user_id: str
    history: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    traits: Dict[str, Any] = field(default_factory=dict)

    def update_history(self, hexagram: str) -> None:
        """Record a consulted hexagram in the user's history."""

        self.history.append(hexagram)

    def set_preference(self, key: str, value: Any) -> None:
        """Update a user's preference."""

        self.preferences[key] = value

    def set_trait(self, key: str, value: Any) -> None:
        """Update a trait in the user's profile."""

        self.traits[key] = value


class PersonalizedHexagramEngine:
    """Generate personalized interpretations based on user profiles."""

    def __init__(self) -> None:
        self._profiles: Dict[str, UserProfile] = {}

    def get_profile(self, user_id: str) -> UserProfile:
        """Retrieve or create a :class:`UserProfile` for ``user_id``."""

        if user_id not in self._profiles:
            self._profiles[user_id] = UserProfile(user_id)
        return self._profiles[user_id]

    def interpret(self, user_id: str, hexagram: str) -> str:
        """Return a personalized interpretation for ``hexagram``.

        The interpretation is adjusted using the user's recorded preferences
        and traits, demonstrating how personalization could be applied.
        """

        profile = self.get_profile(user_id)
        profile.update_history(hexagram)

        base = f"Hexagram {hexagram} signifies change and balance."

        element = profile.preferences.get("element")
        if element == "water":
            advice = "Flow around obstacles and stay adaptable."
        elif element == "fire":
            advice = "Pursue your goals with passion and clarity."
        else:
            advice = "Remain centered and consider all perspectives."

        risk = profile.traits.get("risk_tolerance", "medium")
        if risk == "low":
            advice += " Take cautious steps forward."
        elif risk == "high":
            advice += " Bold moves may bring great rewards."

        return f"{base} {advice}"
