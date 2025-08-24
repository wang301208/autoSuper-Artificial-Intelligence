"""Module for automated self-improvement based on meta tickets and metrics."""

from __future__ import annotations

from pathlib import Path
import csv
import subprocess
from typing import Dict, List, Tuple, Callable

from .ga_config import GAConfig
from .parallel_ga import ParallelGA


class SelfImprovement:
    """Identify bottlenecks, suggest improvements, and trigger actions."""

    def __init__(
        self,
        tickets_dir: Path | str = Path("evolution/meta_tickets"),
        metrics_path: Path | str = Path("evolution/metrics_history.csv"),
        ga_metrics_path: Path | str = Path("evolution/ga_metrics_history.csv"),
    ) -> None:
        self.tickets_dir = Path(tickets_dir)
        self.metrics_path = Path(metrics_path)
        self.ga_metrics_path = Path(ga_metrics_path)

    def _load_metrics(self) -> List[Dict[str, float]]:
        data: List[Dict[str, float]] = []
        try:
            with open(self.metrics_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append({k: float(v) for k, v in row.items()})
        except FileNotFoundError:
            pass
        return data

    def identify_bottlenecks(self) -> List[str]:
        metrics = self._load_metrics()
        if not metrics:
            return ["No metrics available"]
        avg_cpu = sum(m.get("cpu_percent", 0.0) for m in metrics) / len(metrics)
        avg_mem = sum(m.get("memory_percent", 0.0) for m in metrics) / len(metrics)
        bottlenecks: List[str] = []
        if avg_cpu > 80:
            bottlenecks.append("CPU usage high")
        if avg_mem > 80:
            bottlenecks.append("Memory usage high")
        if not bottlenecks:
            bottlenecks.append("No obvious bottlenecks")
        return bottlenecks

    def read_meta_tickets(self) -> List[Tuple[Path, str]]:
        tickets: List[Tuple[Path, str]] = []
        if self.tickets_dir.exists():
            for path in self.tickets_dir.glob("*.md"):
                tickets.append((path, path.read_text()))
        return tickets

    def generate_suggestions(
        self, bottlenecks: List[str], tickets: List[Tuple[Path, str]]
    ) -> List[str]:
        suggestions: List[str] = []
        for b in bottlenecks:
            if b != "No obvious bottlenecks":
                suggestions.append(f"Bottleneck identified: {b}")
        for path, _ in tickets:
            suggestions.append(f"Consider resolving meta ticket: {path.name}")
        if not suggestions:
            suggestions.append("System functioning normally")
        return suggestions

    def trigger_actions(self, tickets: List[Tuple[Path, str]]) -> List[str]:
        actions: List[str] = []
        for path, content in tickets:
            for line in content.splitlines():
                if line.lower().startswith("script:"):
                    script_path = line.split(":", 1)[1].strip()
                    script = Path(script_path)
                    if script.exists():
                        try:
                            subprocess.run(["python", str(script)], check=True)
                            actions.append(f"Executed {script_path}")
                        except Exception as exc:  # pragma: no cover - execution path
                            actions.append(f"Failed to execute {script_path}: {exc}")
                    else:
                        actions.append(f"Script not found: {script_path}")
        return actions

    # Genetic algorithm -------------------------------------------------
    def optimize_with_ga(
        self,
        fitness_fn: Callable[[List[float]], float] | None = None,
        config: GAConfig | None = None,
    ) -> Dict[str, float | List[float]]:
        """Run the parallel GA and record generation/time metrics."""

        if fitness_fn is None:
            fitness_fn = lambda x: -sum((v - 0.5) ** 2 for v in x)
        cfg = config or GAConfig.from_env()
        ga = ParallelGA(fitness_fn, cfg)
        best, best_fit, history = ga.run()

        self._record_ga_history(history)
        return {"best_individual": best, "best_fitness": best_fit}

    def _record_ga_history(self, history: List[Dict[str, float]]) -> None:
        """Append GA generation/time metrics to CSV."""

        self.ga_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.ga_metrics_path.exists()
        with open(self.ga_metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["generation", "elapsed_time", "best_fitness"]
            )
            if write_header:
                writer.writeheader()
            for row in history:
                writer.writerow(row)

    def run(self) -> Dict[str, List[str]]:
        tickets = self.read_meta_tickets()
        bottlenecks = self.identify_bottlenecks()
        suggestions = self.generate_suggestions(bottlenecks, tickets)
        actions = self.trigger_actions(tickets)
        ga_result = self.optimize_with_ga()
        return {
            "bottlenecks": bottlenecks,
            "suggestions": suggestions,
            "actions": actions,
            "ga_result": ga_result,
        }
