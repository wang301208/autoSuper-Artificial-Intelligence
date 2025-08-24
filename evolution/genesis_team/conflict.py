"""Conflict detection and resolution for Genesis team."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ConflictResolver:
    """Detects conflicts in agent outputs and coordinates resolution."""

    history: List[str] = field(default_factory=list)

    def detect(self, output: str) -> bool:
        """Return True if the output contains signs of conflict."""

        lowered = output.lower()
        keywords = ["conflict", "version mismatch", "overlap", "error"]
        return any(keyword in lowered for keyword in keywords)

    def resolve(self, agent_name: str, logs: Dict[str, str]) -> str:
        """Resolve detected conflicts by rolling back or merging.

        Parameters
        ----------
        agent_name
            Name of the agent whose output was most recently produced.
        logs
            Mapping of agent names to their output logs.

        Returns
        -------
        str
            Decision summary, either a merge or rollback description.
        """

        output = logs[agent_name]
        if self.detect(output):
            logs[agent_name] = f"ROLLED BACK: {output}"
            decision = f"{agent_name}: rollback"
        else:
            decision = f"{agent_name}: merge"
        self.history.append(decision)
        return decision
