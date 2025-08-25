from __future__ import annotations

from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor
import os
import asyncio

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
        vector_type: str = "text",
    ) -> None:
        """Store a skill and optionally index its embedding."""
        self.library.add_skill(name, code, metadata)
        self.graph.add_node(name, EntityType.SKILL, **metadata)
        if self.index and embedding is not None:
            self.index.add(name, embedding, metadata, vector_type=vector_type)

    def search(
        self,
        embedding: List[float],
        n_results: int = 1,
        vector_type: str = "text",
        return_content: bool = False,
        max_workers: int | None = None,
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
        max_workers: int | None
            Maximum number of threads to use when fetching content. ``None``
            uses ``min(n_results, os.cpu_count() or 1)``. ``1`` disables
            parallelism.
        """
        if not self.index:
            raise RuntimeError("Vector index not available")
        result = self.index.query(embedding, n_results, vector_type=vector_type)
        ids = result.get("ids", [[]])[0]
        if return_content:
            if len(ids) <= 1 or (max_workers is not None and max_workers <= 1):
                return [self.get_skill(name)[0] for name in ids]

            workers = min(len(ids), max_workers or (os.cpu_count() or 1))

            def _load(name: str) -> str:
                return self.get_skill(name)[0]

            with ThreadPoolExecutor(max_workers=workers) as executor:
                return list(executor.map(_load, ids))
        return ids

    def get_skill(self, name: str):
        return asyncio.run(self.library.get_skill(name))

    def list_skills(self) -> List[str]:
        return self.library.list_skills()

    # -- Knowledge graph helpers ---------------------------------------------
    def relate_skills(self, source: str, target: str) -> None:
        """Create a relation between two skills in the knowledge graph."""

        self.graph.add_edge(source, target, RelationType.RELATED_TO)

    def query_graph(self, **kwargs):
        """Proxy to the underlying graph store's query method."""

        return self.graph.query(**kwargs)
