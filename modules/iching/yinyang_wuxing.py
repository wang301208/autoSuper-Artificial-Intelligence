"""Yin-Yang transformation and Five Elements interactions."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class YinYangFiveElements:
    """Model Yin-Yang conversion and Five Elements relationships."""

    # Mapping of yin/yang to their opposite values
    yin_yang_map = {"yin": "yang", "yang": "yin"}

    # Five Elements generating cycle (sheng)
    generating_cycle = {
        "wood": "fire",
        "fire": "earth",
        "earth": "metal",
        "metal": "water",
        "water": "wood",
    }

    # Five Elements overcoming cycle (ke)
    overcoming_cycle = {
        "wood": "earth",
        "earth": "water",
        "water": "fire",
        "fire": "metal",
        "metal": "wood",
    }

    @staticmethod
    def transform_yinyang(value: str) -> str:
        """Convert a yin/yang value to its opposite.

        Args:
            value: Either ``"yin"`` or ``"yang"``.

        Returns:
            The opposite yin/yang value.

        Raises:
            ValueError: If ``value`` is not ``"yin"`` or ``"yang"``.
        """

        try:
            return YinYangFiveElements.yin_yang_map[value.lower()]
        except KeyError as exc:
            raise ValueError("Value must be 'yin' or 'yang'") from exc

    @classmethod
    def element_interaction(cls, element_a: str, element_b: str) -> str:
        """Determine the interaction between two Five Elements.

        Args:
            element_a: The acting element.
            element_b: The element acted upon.

        Returns:
            One of ``"generates"``, ``"generated_by"``, ``"overcomes"``,
            ``"overcome_by"`` or ``"neutral"``.

        Raises:
            ValueError: If an element is not one of the five.
        """

        a = element_a.lower()
        b = element_b.lower()
        elements = cls.generating_cycle.keys()
        if a not in elements or b not in elements:
            raise ValueError("Invalid element")

        if cls.generating_cycle[a] == b:
            return "generates"
        if cls.generating_cycle[b] == a:
            return "generated_by"
        if cls.overcoming_cycle[a] == b:
            return "overcomes"
        if cls.overcoming_cycle[b] == a:
            return "overcome_by"
        return "neutral"
