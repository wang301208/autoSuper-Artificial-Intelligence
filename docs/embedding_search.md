# Embedding Search Strategy

To determine how to represent documents for similarity search, we benchmarked two
approaches:

* **Summary-based** – encode a summary of each document and compare queries
  against those summary embeddings.
* **Weighted-average** – compute embeddings for each chunk of a document and use
  the token-length weighted average as the document representation.

The benchmark (see `modules/benchmark/embedding_search_benchmark.py`) evaluated retrieval
accuracy on a small corpus of animal facts. Results:

```
Summary-based accuracy: 3/3
Weighted-average accuracy: 3/3
Chosen approach: summary-based
```

Both strategies performed similarly on this dataset; summary-based search is
retained as the default for its simplicity and efficiency.
