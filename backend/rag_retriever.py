"""Simple Retrieval Augmented Generation helper."""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, List, Tuple

from capability.librarian import Librarian


class RAGRetriever:
    """Perform retrieval augmented generation using the skill library."""

    def __init__(self, librarian: Librarian) -> None:
        self.librarian = librarian
        # Reusable buffer for retrieved documents to reduce temporary object creation.
        self._docs_buffer: List[str] = []

    @lru_cache(maxsize=128)
    def _cached_search(
        self, embedding_key: Tuple[float, ...], n_results: int, vector_type: str
    ) -> Tuple[str, ...]:
        """Cache search results for repeated queries."""
        return tuple(
            self.librarian.search(
                list(embedding_key),
                n_results=n_results,
                vector_type=vector_type,
                return_content=True,
            )
        )

    def generate(
        self,
        prompt: str,
        query_embedding: List[float],
        llm_callable: Callable[[str], str],
        n_results: int = 3,
        vector_type: str = "text",
    ) -> str:
        """Generate LLM output with retrieved context.

        Parameters
        ----------
        prompt: str
            The user prompt/question.
        query_embedding: List[float]
            Embedding of the query used for similarity search.
        llm_callable: Callable[[str], str]
            Function that takes the final prompt and returns generated text.
        n_results: int
            Number of documents to retrieve.
        vector_type: str
            Vector space to query (e.g. ``"text"`` or ``"image"``).
        """
        docs = list(self._cached_search(tuple(query_embedding), n_results, vector_type))
        self._docs_buffer.clear()
        self._docs_buffer.extend(docs)
        context = "\n".join(self._docs_buffer)
        final_prompt = f"{context}\n\n{prompt}" if context else prompt
        return llm_callable(final_prompt)

