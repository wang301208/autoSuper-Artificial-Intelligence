from __future__ import annotations

"""Hexagram interpretation engine using predefined analysis dimensions."""

from typing import Iterable, Dict

from .analysis_dimensions import ANALYSIS_DIMENSIONS, DimensionRule, get_dimension_rule


class HexagramEngine:
    """Provide interpretations for hexagrams across multiple analysis dimensions."""

    def interpret_hexagram(
        self, hexagram: str, dimensions: Iterable[str]
    ) -> Dict[str, Dict[str, str]]:
        """Interpret ``hexagram`` within the specified ``dimensions``.

        Parameters
        ----------
        hexagram: str
            Six-character string representing the hexagram (``1`` for yang, ``0`` for yin).
        dimensions: Iterable[str]
            Names of the analysis dimensions to evaluate.

        Returns
        -------
        Dict[str, Dict[str, str]]
            Mapping of dimension name to a dictionary containing ``interpretation`` and
            ``advice``.
        """

        results: Dict[str, Dict[str, str]] = {}
        for dimension in dimensions:
            if dimension not in ANALYSIS_DIMENSIONS:
                raise KeyError(f"Unknown dimension: {dimension}")
            rule: DimensionRule = get_dimension_rule(dimension, hexagram)
            results[dimension] = {
                "interpretation": rule.interpretation,
                "advice": rule.advice,
            }
        return results
