from __future__ import annotations

from typing import Dict, List, Any

try:
    import chromadb
    from chromadb.config import Settings
except Exception as exc:  # pragma: no cover - chromadb is optional in tests
    chromadb = None
    Settings = None


class VectorIndex:
    """Wrapper around ChromaDB for skill embeddings."""

    def __init__(self, persist_directory: str | None = None) -> None:
        if chromadb is None:
            raise ImportError("chromadb is required for VectorIndex")
        self.client = chromadb.Client(Settings(persist_directory=persist_directory))
        self.collection = self.client.get_or_create_collection("skills")

    def add(self, skill_id: str, embedding: List[float], metadata: Dict[str, Any] | None = None) -> None:
        self.collection.add(ids=[skill_id], embeddings=[embedding], metadatas=[metadata or {}])

    def query(self, embedding: List[float], n_results: int = 1) -> Dict[str, Any]:
        return self.collection.query(query_embeddings=[embedding], n_results=n_results)
