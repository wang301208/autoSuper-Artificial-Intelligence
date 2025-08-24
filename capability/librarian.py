from __future__ import annotations

from typing import Dict, List

from .skill_library import SkillLibrary

try:
    from .vector_index import VectorIndex
except Exception:  # pragma: no cover - vector index optional
    VectorIndex = None


class Librarian:
    """High level API for skill storage and retrieval."""

    def __init__(self, repo_path: str, persist_directory: str | None = None) -> None:
        self.library = SkillLibrary(repo_path)
        self.index = None
        if VectorIndex is not None:
            try:
                self.index = VectorIndex(persist_directory)
            except Exception:
                self.index = None

    def add_skill(
        self,
        name: str,
        code: str,
        metadata: Dict,
        embedding: List[float] | None = None,
        vector_type: str = "text",
    ) -> None:
        """Store a skill and optionally index its embedding."""
        self.library.add_skill(name, code, metadata)
        if self.index and embedding is not None:
            self.index.add(name, embedding, metadata, vector_type=vector_type)

    def search(
        self,
        embedding: List[float],
        n_results: int = 1,
        vector_type: str = "text",
        return_content: bool = False,
    ) -> List[str]:
        """Search skills by embedding.

        Parameters
        ----------
        embedding: List[float]
            The embedding to query with.
        n_results: int
            Number of results to retrieve.
        vector_type: str
            Which vector space to query in.
        return_content: bool
            When ``True`` return the document contents instead of IDs.
        """
        if not self.index:
            raise RuntimeError("Vector index not available")
        result = self.index.query(embedding, n_results, vector_type=vector_type)
        ids = result.get("ids", [[]])[0]
        if return_content:
            return [self.get_skill(name)[0] for name in ids]
        return ids

    def get_skill(self, name: str):
        return self.library.get_skill(name)

    def list_skills(self) -> List[str]:
        return self.library.list_skills()
