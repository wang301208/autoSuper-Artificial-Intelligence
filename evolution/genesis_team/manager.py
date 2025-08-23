"""Manager coordinating Genesis team agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from .sentinel import Sentinel
from .archaeologist import Archaeologist
from .tdd_dev import TDDDeveloper
from .qa import QA


@dataclass
class GenesisTeamManager:
    """Orchestrates the Genesis team agents.

    The manager instantiates each agent and provides a :meth:`run` method
    which executes them sequentially, collecting their outputs. The order of
    execution is Sentinel -> Archaeologist -> TDDDeveloper -> QA. Agent
    instances can be provided for testing purposes; otherwise defaults are
    created.
    """

    sentinel: Sentinel = field(default_factory=Sentinel)
    archaeologist: Archaeologist = field(default_factory=Archaeologist)
    tdd_dev: TDDDeveloper = field(default_factory=TDDDeveloper)
    qa: QA = field(default_factory=QA)

    def run(self) -> Dict[str, str]:
        """Execute all agents and return their logs.

        Returns
        -------
        Dict[str, str]
            A dictionary mapping agent names to their output. The insertion
            order reflects the execution order.
        """

        logs: Dict[str, str] = {}
        logs["sentinel"] = self.sentinel.perform()
        logs["archaeologist"] = self.archaeologist.perform()
        logs["tdd_developer"] = self.tdd_dev.perform()
        logs["qa"] = self.qa.perform()
        return logs
