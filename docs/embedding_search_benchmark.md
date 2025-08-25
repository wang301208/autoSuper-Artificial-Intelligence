
# Embedding Search Benchmark

A small benchmark compared two strategies for representing a memory item during
vector search:

* **Summary embedding** – embedding of the textual summary of a memory item.
* **Weighted average** – weighted mean of embeddings for each chunk, with the
  weight proportional to chunk length.

Using a tiny dataset of three sample sentences and semantically related queries
(see `benchmarks/embedding_search.py`) both strategies retrieved the correct
item for all three queries (`summary correct 3`, `weighted correct 3`).  Given
its comparable accuracy and ability to capture details without relying on an
additional summarization step, the weighted-average strategy was chosen as the
default representation stored in `MemoryItem.e_summary`.
