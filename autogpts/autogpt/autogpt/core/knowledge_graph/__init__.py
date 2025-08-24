"""Knowledge graph utilities for AutoGPT."""

from .ontology import EntityType, RelationType
from .graph_store import GraphStore, get_graph_store, query_graph

__all__ = [
    "EntityType",
    "RelationType",
    "GraphStore",
    "get_graph_store",
    "query_graph",
]
