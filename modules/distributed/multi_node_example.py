"""Example demonstrating multi-node task execution using DistributedBrainNode.

The script spawns a master node and a number of worker processes that connect to
it.  It measures the time required to process a set of tasks with different
numbers of workers to provide a rough indication of scalability.  The example
also includes a simple fault-tolerance demonstration where one worker raises an
exception while others continue processing.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from typing import List

# Allow the example to run both as a script and as a module
if __package__ is None or __package__ == "":
    import os
    import sys

    sys.path.append(os.path.dirname(__file__))
    from distributed_brain_node import DistributedBrainNode
else:  # pragma: no cover - imported when executed as a package
    from .distributed_brain_node import DistributedBrainNode


def square(x: int) -> int:
    """Example task handler that squares its input after a short delay."""

    time.sleep(0.1)
    return x * x


def crashing_worker(address: tuple[str, int], authkey: bytes) -> None:
    """Process target that crashes to simulate a faulty node."""

    raise RuntimeError("simulated worker failure")


def run_demo() -> None:
    tasks: List[int] = list(range(10))
    authkey = b"autogpt"
    base_port = 50000

    print("--- Scalability demo ---")
    for idx, workers in enumerate((1, 2, 4)):
        address = ("localhost", base_port + idx)
        master = DistributedBrainNode(address, authkey)
        master.start_master(tasks)

        start = time.time()
        procs = [
            mp.Process(
                target=DistributedBrainNode(address, authkey).run_worker,
                args=(square,),
            )
            for _ in range(workers)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        duration = time.time() - start

        results = master.gather_results(len(tasks))
        master.shutdown()
        print(f"{workers} workers -> {duration:.2f}s, results={results}")

    print("\n--- Fault tolerance demo ---")
    address = ("localhost", base_port + 10)
    master = DistributedBrainNode(address, authkey)
    master.start_master(tasks)
    procs = [
        mp.Process(target=DistributedBrainNode(address, authkey).run_worker, args=(square,)),
        mp.Process(target=crashing_worker, args=(address, authkey)),
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    results = master.gather_results(len(tasks))
    master.shutdown()
    print(
        f"results with flaky worker (exit {procs[1].exitcode}): {results}"
    )


if __name__ == "__main__":
    run_demo()
