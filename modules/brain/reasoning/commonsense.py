"""Common-sense reasoning via ConceptNet."""
from __future__ import annotations

import re
from typing import Dict, List

import requests


class CommonSenseReasoner:
    """Interface to external common-sense knowledge graphs like ConceptNet.

    This minimal implementation queries the public ConceptNet API to retrieve
    edges related to tokens extracted from natural language text. Returned
    results include a textual conclusion and a normalized confidence score.
    """

    api_base = "https://api.conceptnet.io"

    def __init__(self, language: str = "en") -> None:
        self.language = language

    def _query_conceptnet(self, term: str) -> List[Dict[str, float]]:
        """Query ConceptNet for edges related to ``term``.

        Parameters
        ----------
        term: str
            Concept term to query in ConceptNet.

        Returns
        -------
        list of dict
            A list of reasoning conclusions with confidence scores.
        """
        url = f"{self.api_base}/query?node=/c/{self.language}/{term}&limit=50"
        try:
            data = requests.get(url, timeout=5).json()
        except requests.RequestException:
            return []

        results: List[Dict[str, float]] = []
        for edge in data.get("edges", []):
            rel = edge["rel"]["label"]
            start = edge["start"]["label"]
            end = edge["end"]["label"]
            weight = float(edge.get("weight", 1.0))
            conclusion = f"{start} {rel} {end}"
            confidence = min(weight / 10.0, 1.0)
            results.append({"conclusion": conclusion, "confidence": confidence})
        return results

    def infer(self, text: str) -> List[Dict[str, float]]:
        """Infer common-sense knowledge from natural language text.

        Parameters
        ----------
        text: str
            Input natural language string.

        Returns
        -------
        list of dict
            Each element contains ``conclusion`` and ``confidence``.
        """
        terms = {t for t in re.findall(r"\w+", text.lower()) if t}
        results: List[Dict[str, float]] = []
        for term in terms:
            results.extend(self._query_conceptnet(term))
        return results
