"""A minimal in-memory graph store for skills, tasks and their relations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .ontology import EntityType, RelationType


@dataclass
class Node:
    id: str
    type: EntityType
    properties: Dict[str, object] = field(default_factory=dict)


@dataclass
class Edge:
    source: str
    target: str
    type: RelationType
    properties: Dict[str, object] = field(default_factory=dict)


class GraphStore:
    """Simple in-memory graph storage."""

    def __init__(self) -> None:
        self._nodes: Dict[str, Node] = {}
        self._edges: List[Edge] = []

    # -- Node/edge management -------------------------------------------------
    def add_node(
        self, node_id: str, entity_type: EntityType, **properties: object
    ) -> None:
        self._nodes[node_id] = Node(node_id, entity_type, properties)

    def add_edge(
        self,
        source: str,
        target: str,
        relation_type: RelationType,
        **properties: object,
    ) -> None:
        self._edges.append(Edge(source, target, relation_type, properties))

    # -- Query API ------------------------------------------------------------
    def query(
        self,
        node_id: Optional[str] = None,
        entity_type: Optional[EntityType] = None,
        relation_type: Optional[RelationType] = None,
    ) -> Dict[str, List]:
        """Query nodes and edges in the graph.

        Args:
            node_id: If provided, limit results to this node and its edges.
            entity_type: Filter nodes by this entity type.
            relation_type: Filter edges by this relation type.
        """

        nodes = [n for n in self._nodes.values() if not entity_type or n.type == entity_type]
        if node_id:
            nodes = [n for n in nodes if n.id == node_id]

        edges = self._edges
        if node_id:
            edges = [e for e in edges if e.source == node_id or e.target == node_id]
        if relation_type:
            edges = [e for e in edges if e.type == relation_type]

        return {
            "nodes": nodes,
            "edges": edges,
        }


_GLOBAL_GRAPH_STORE: GraphStore | None = None


def get_graph_store() -> GraphStore:
    """Return a module-wide graph store instance."""

    global _GLOBAL_GRAPH_STORE
    if _GLOBAL_GRAPH_STORE is None:
        _GLOBAL_GRAPH_STORE = GraphStore()
    return _GLOBAL_GRAPH_STORE


def query_graph(**kwargs):
    """Convenience wrapper around :meth:`GraphStore.query`."""

    return get_graph_store().query(**kwargs)
