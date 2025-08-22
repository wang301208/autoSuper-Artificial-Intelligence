"""Strategist agent that extracts strategic principles from task logs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from . import Agent


class Strategist(Agent):
    """Aggregates logs, derives principles, and writes them to the charter."""

    def __init__(self, charter_dir: Path | str = Path("governance/charter")) -> None:
        self.charter_dir = Path(charter_dir)
        self.charter_dir.mkdir(parents=True, exist_ok=True)

    def perform(self, logs: Iterable[Path] | None = None) -> Path:
        lines: List[str] = []
        for log in logs or []:
            path = Path(log)
            if path.exists():
                lines.extend(path.read_text().splitlines())

        principles = self._extract_principles(lines)
        charter_path = self.charter_dir / "strategic_principles.md"
        charter_path.write_text("\n".join(principles) + "\n")
        return charter_path

    def _extract_principles(self, lines: List[str]) -> List[str]:
        unique: List[str] = []
        for line in lines:
            line = line.strip()
            if line and line not in unique:
                unique.append(f"- {line}")
        if not unique:
            unique.append("- No actionable principles found.")
        return unique
