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
    ) -> None:
        """Store a skill and optionally index its embedding."""
        self.library.add_skill(name, code, metadata)
        if self.index and embedding is not None:
            self.index.add(name, embedding, metadata)

    def search(self, embedding: List[float], n_results: int = 1) -> List[str]:
        """Search skills by embedding."""
        if not self.index:
            raise RuntimeError("Vector index not available")
        result = self.index.query(embedding, n_results)
        return result.get("ids", [[]])[0]

    def get_skill(self, name: str):
        return self.library.get_skill(name)

    def list_skills(self) -> List[str]:
        return self.library.list_skills()
