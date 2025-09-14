from __future__ import annotations

"""Definitions for I Ching analysis dimensions and their interpretation rules."""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class DimensionRule:
    """Represents interpretation rules for a hexagram within a dimension.

    Attributes
    ----------
    interpretation: str
        The meaning derived from the hexagram for the given dimension.
    advice: str
        Actionable guidance associated with the interpretation.
    """

    interpretation: str
    advice: str


# Mapping of dimension name -> hexagram -> DimensionRule
# Hexagrams are represented as six-character strings of ``1`` (yang) and ``0`` (yin).
ANALYSIS_DIMENSIONS: Dict[str, Dict[str, DimensionRule]] = {
    "career": {
        "111111": DimensionRule(
            "Great potential for leadership", "Take charge of projects."),
        "000000": DimensionRule(
            "Need for support and collaboration", "Seek guidance from mentors."),
    },
    "relationship": {
        "111111": DimensionRule(
            "Strong personal drive may overshadow others", "Practice empathy."),
        "000000": DimensionRule(
            "Stable and nurturing environment", "Foster open communication."),
    },
    "fortune": {
        "111111": DimensionRule(
            "Favorable circumstances ahead", "Capitalize on momentum."),
        "000000": DimensionRule(
            "Challenges may arise", "Remain patient and prepared."),
    },
}


def get_dimension_rule(dimension: str, hexagram: str) -> DimensionRule:
    """Retrieve the rule for a given dimension and hexagram.

    Parameters
    ----------
    dimension: str
        Name of the analysis dimension.
    hexagram: str
        Six-character hexagram key composed of ``1`` and ``0``.

    Returns
    -------
    DimensionRule
        The corresponding rule.

    Raises
    ------
    KeyError
        If either the dimension or hexagram is not defined.
    """

    try:
        return ANALYSIS_DIMENSIONS[dimension][hexagram]
    except KeyError as exc:  # pragma: no cover - defensive error path
        raise KeyError(
            f"Unknown dimension '{dimension}' or hexagram '{hexagram}'" 
        ) from exc
