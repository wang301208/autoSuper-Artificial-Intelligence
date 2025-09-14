# Embedding Search Strategy

To determine how to represent documents for similarity search, we benchmarked two
approaches:

* **Summary-based** – encode a summary of each document and compare queries
  against those summary embeddings.
* **Weighted-average** – compute embeddings for each chunk of a document and use
  the token-length weighted average as the document representation.

The benchmark (`benchmarks/memory_embedding_strategy.py`) evaluated recall and
latency on randomly generated embeddings. Results:

```
Summary recall: 12%, latency: 0.0132s
Weighted-average recall: 12%, latency: 0.0130s
Chosen approach: weighted-average
```

Both strategies produced the same recall on this dataset, but the weighted
average was marginally faster, so it is the default. You can switch strategies
via the `memory_embedding_strategy` configuration option.
