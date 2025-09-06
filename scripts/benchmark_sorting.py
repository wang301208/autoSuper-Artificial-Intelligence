import os
import random
import time
from pathlib import Path

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithms.sorting.basic.bubble_sort import BubbleSort
from algorithms.sorting.basic.quick_sort import QuickSort

RESULT_FILE = Path(__file__).resolve().parent.parent / "benchmarks" / "sorting_benchmark.txt"


def benchmark() -> None:
    sizes = [50, 100, 200]
    lines = ["size,bubble_sort_ms,quick_sort_ms"]
    for n in sizes:
        data = random.sample(range(n * 2), n)
        start = time.perf_counter()
        BubbleSort().execute(data)
        bubble_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        QuickSort().execute(data)
        quick_ms = (time.perf_counter() - start) * 1000
        lines.append(f"{n},{bubble_ms:.3f},{quick_ms:.3f}")
    RESULT_FILE.parent.mkdir(exist_ok=True)
    RESULT_FILE.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    benchmark()
