from __future__ import annotations

"""Real-time metrics collection utilities."""

from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Optional
import time

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil optional
    psutil = None  # type: ignore

from .bottleneck import BottleneckDetector


@dataclass
class MetricEvent:
    """Container for metrics from a single operation."""

    module: str
    latency: float
    energy: float
    throughput: float
    timestamp: float


class RealTimeMetricsCollector:
    """Collects latency, energy consumption, and throughput in real time.

    The collector can be used as::

        collector.start("module")
        ... do work ...
        event = collector.end("module")
    """

    def __init__(self, detector: Optional[BottleneckDetector] = None) -> None:
        self._detector = detector
        self._events: List[MetricEvent] = []
        self._starts: Dict[str, tuple[float, float]] = {}
        self._counts: Dict[str, int] = defaultdict(int)
        self._process = psutil.Process() if psutil else None

    # ------------------------------------------------------------------
    def start(self, module: str) -> None:
        """Mark the start of an operation for ``module``."""
        start_time = time.perf_counter()
        start_cpu = 0.0
        if self._process is not None:
            cpu = self._process.cpu_times()
            start_cpu = cpu.user + cpu.system
        self._starts[module] = (start_time, start_cpu)

    # ------------------------------------------------------------------
    def end(self, module: str, items: int = 1) -> MetricEvent:
        """Finish an operation for ``module`` and record metrics."""
        start_time, start_cpu = self._starts.pop(module, (time.perf_counter(), 0.0))
        end_time = time.perf_counter()
        latency = end_time - start_time

        energy = 0.0
        if self._process is not None:
            cpu = self._process.cpu_times()
            end_cpu = cpu.user + cpu.system
            energy = max(end_cpu - start_cpu, 0.0)

        self._counts[module] += items
        throughput = 0.0
        if latency > 0:
            throughput = items / latency

        event = MetricEvent(
            module=module,
            latency=latency,
            energy=energy,
            throughput=throughput,
            timestamp=end_time,
        )
        self._events.append(event)

        if self._detector is not None:
            self._detector.record(module, latency)

        return event

    # ------------------------------------------------------------------
    def events(self) -> List[MetricEvent]:
        """Return all recorded metric events."""
        return list(self._events)

    # ------------------------------------------------------------------
    def print_dashboard(self) -> None:
        """Print a simple dashboard with average metrics per module."""
        if not self._events:
            print("No metrics collected yet")
            return

        stats: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        counts: Dict[str, int] = defaultdict(int)
        for event in self._events:
            stats[event.module]["latency"] += event.latency
            stats[event.module]["energy"] += event.energy
            stats[event.module]["throughput"] += event.throughput
            counts[event.module] += 1

        header = f"{'Module':<20}{'Avg Latency':>15}{'Avg Energy':>15}{'Avg Thpt':>15}"
        print(header)
        print("-" * len(header))
        for module, s in stats.items():
            n = counts[module]
            line = (
                f"{module:<20}"
                f"{s['latency']/n:>15.4f}"
                f"{s['energy']/n:>15.4f}"
                f"{s['throughput']/n:>15.4f}"
            )
            print(line)
