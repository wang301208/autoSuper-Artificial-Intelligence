"""Strategist agent that extracts strategic principles from task logs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple
from datetime import datetime
from threading import Timer
from heapq import heappop, heappush
import json

from capability.meta_skill import META_SKILL_STRATEGY_EVOLUTION
from capability.skill_library import SkillLibrary

from . import Agent


class Strategist(Agent):
    """Aggregates logs, derives principles, and writes them to the charter."""

    def __init__(
        self,
        charter_dir: Path | str = Path("governance/charter"),
        metrics_dir: Path | str = Path("governance/metrics/strategist"),
        skill_repo: Path | str = Path("."),
    ) -> None:
        self.charter_dir = Path(charter_dir)
        self.charter_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        # Local library of meta-skills consulted during analysis
        self.library = SkillLibrary(skill_repo)

    def perform(self, logs: Iterable[Path] | None = None) -> Path:
        # Load strategist reasoning template to guide log analysis
        template, _meta = self.library.get_skill(META_SKILL_STRATEGY_EVOLUTION)
        lines: List[str] = template.splitlines()
        for log in logs or []:
            path = Path(log)
            if path.exists():
                lines.extend(path.read_text().splitlines())

        principles = self._extract_principles(lines)
        charter_path = self.charter_dir / "strategic_principles.md"
        charter_path.write_text("\n".join(principles) + "\n")

        # Determine an executable action sequence using A* planning
        actions = self._plan_actions(principles)
        plan_path = self.charter_dir / "strategic_plan.md"
        plan_path.write_text("\n".join(f"- {step}" for step in actions) + "\n")

        # Reflect on recent performance metrics
        self.reflect(logs=logs, principles=principles)
        return plan_path

    def reflect(
        self,
        logs: Iterable[Path] | None = None,
        principles: List[str] | None = None,
    ) -> Path:
        """Compute performance metrics from historical task logs."""
        lines: List[str] = []
        for log in logs or []:
            path = Path(log)
            if path.exists():
                lines.extend(path.read_text().splitlines())

        successes = sum(1 for line in lines if "success" in line.lower())
        failures = sum(1 for line in lines if "fail" in line.lower())
        total = successes + failures
        success_rate = successes / total if total else 0.0

        existing = sorted(self.metrics_dir.glob("*.json"))
        prev_success = 0.0
        prev_principles = 0
        if existing:
            prev_data = json.loads(existing[-1].read_text())
            prev_success = prev_data.get("success_rate", 0.0)
            prev_principles = prev_data.get("principles_count", 0)

        success_rate_change = success_rate - prev_success
        principles_count = len(principles or [])
        principle_derivation_velocity = principles_count - prev_principles

        data = {
            "successes": successes,
            "failures": failures,
            "success_rate": success_rate,
            "success_rate_change": success_rate_change,
            "principles_count": principles_count,
            "principle_derivation_velocity": principle_derivation_velocity,
        }

        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        metrics_path = self.metrics_dir / f"{timestamp}.json"
        metrics_path.write_text(json.dumps(data, indent=2))

        # Determine if performance has stagnated and a meta-upgrade is needed
        if existing and (
            success_rate_change <= 0 or principle_derivation_velocity <= 0
        ):
            reasons: List[str] = []
            if success_rate_change <= 0:
                reasons.append(
                    f"Success rate stagnated at {success_rate:.2%}"
                )
            if principle_derivation_velocity <= 0:
                reasons.append(
                    "No new strategic principles were derived"
                )
            self.request_meta_upgrade("; ".join(reasons))

        return metrics_path

    def schedule_reflection(
        self, interval_hours: int = 24, logs: Iterable[Path] | None = None
    ) -> Timer:
        """Schedule recurring reflection runs via a simple timer."""

        def _run() -> None:
            self.reflect(logs=logs)
            self.schedule_reflection(interval_hours, logs)

        timer = Timer(interval_hours * 3600, _run)
        timer.daemon = True
        timer.start()
        return timer

    def request_meta_upgrade(self, reason: str) -> Path:
        """Create a meta-upgrade ticket summarizing current limitations."""
        tickets_dir = Path("evolution/meta_tickets")
        tickets_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        ticket_path = tickets_dir / f"{timestamp}.md"
        content = (
            f"# Meta Ticket\n\n"
            f"- Created: {timestamp}\n\n"
            f"## Limitation\n{reason}\n\n"
            "## Desired Improvements\n\n"
            "- [ ] Detail proposed improvement\n"
        )
        ticket_path.write_text(content)
        return ticket_path

    def _extract_principles(self, lines: List[str]) -> List[str]:
        unique: List[str] = []
        for line in lines:
            line = line.strip()
            if line and line not in unique:
                unique.append(f"- {line}")
        if not unique:
            unique.append("- No actionable principles found.")
        return unique

    def _plan_actions(self, principles: List[str]) -> List[str]:
        """Generate an ordered action plan from principles using A* search."""

        def heuristic(idx: int) -> int:
            # Remaining principles act as a simple distance heuristic
            return len(principles) - idx

        frontier: List[Tuple[int, int, List[str]]] = []
        heappush(frontier, (heuristic(0), 0, []))
        visited = set()

        while frontier:
            _score, idx, path = heappop(frontier)
            if idx == len(principles):
                return path
            if idx in visited:
                continue
            visited.add(idx)

            principle_text = principles[idx].lstrip("- ").strip()
            step = f"Act on: {principle_text}"
            new_path = path + [step]
            cost = len(new_path)
            heappush(frontier, (cost + heuristic(idx + 1), idx + 1, new_path))

        return []
