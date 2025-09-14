"""Example demonstrating spike train storage and querying.

This script writes random spike trains using :mod:`modules.brain.data_manager`
components and measures compression ratio and retrieval speed.
"""
from __future__ import annotations

import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from modules.brain.data_manager import (
    DistributedNeuralStorage,
    NeuralDataIndexer,
    SpikeDataCompressor,
    query_neural_patterns,
    store_spike_train,
)


def run_example() -> None:
    rng = np.random.default_rng(0)
    spikes = np.cumsum(rng.integers(1, 10, size=1000))
    pattern = spikes[100:105]

    with TemporaryDirectory() as tmpdir:
        storage = DistributedNeuralStorage([Path(tmpdir) / "n0", Path(tmpdir) / "n1"])
        compressor = SpikeDataCompressor()
        indexer = NeuralDataIndexer(window_size=len(pattern))

        ratio = store_spike_train("neuron0", spikes, compressor, indexer, storage)
        start = time.perf_counter()
        matches = query_neural_patterns(pattern, indexer, storage, compressor)
        elapsed = time.perf_counter() - start

        print(f"Compression ratio: {ratio:.2f}")
        print(f"Query matches: {matches}, time: {elapsed:.4f}s")


if __name__ == "__main__":
    run_example()
