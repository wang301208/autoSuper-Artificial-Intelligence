from __future__ import annotations

from typing import Dict, List

try:
    from autogpt.core.knowledge_graph import (
        EntityType,
        RelationType,
        get_graph_store,
    )
except Exception:  # pragma: no cover - fallback when package not installed
    from autogpts.autogpt.autogpt.core.knowledge_graph import (  # type: ignore
        EntityType,
        RelationType,
        get_graph_store,
    )

from .skill_library import SkillLibrary

try:
    from .vector_index import VectorIndex
except Exception:  # pragma: no cover - vector index optional
    VectorIndex = None


class Librarian:
    """High level API for skill storage and retrieval."""

    def __init__(
        self,
        repo_path: str,
        persist_directory: str | None = None,
        graph_store=None,
    ) -> None:
        self.library = SkillLibrary(repo_path)
        self.graph = graph_store or get_graph_store()
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
        self.graph.add_node(name, EntityType.SKILL, **metadata)
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

    # -- Knowledge graph helpers ---------------------------------------------
    def relate_skills(self, source: str, target: str) -> None:
        """Create a relation between two skills in the knowledge graph."""

        self.graph.add_edge(source, target, RelationType.RELATED_TO)

    def query_graph(self, **kwargs):
        """Proxy to the underlying graph store's query method."""

        return self.graph.query(**kwargs)
