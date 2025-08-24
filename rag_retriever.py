"""Simple Retrieval Augmented Generation helper."""
from __future__ import annotations

from typing import Callable, List

from capability.librarian import Librarian


class RAGRetriever:
    """Perform retrieval augmented generation using the skill library."""

    def __init__(self, librarian: Librarian) -> None:
        self.librarian = librarian

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
        docs = self.librarian.search(
            query_embedding,
            n_results=n_results,
            vector_type=vector_type,
            return_content=True,
        )
        context = "\n".join(docs)
        final_prompt = f"{context}\n\n{prompt}" if context else prompt
        return llm_callable(final_prompt)
