"""Knowledge integration utilities.

This package provides components for integrating multiple knowledge sources
into a unified representation that can be queried by other parts of the
system.
"""

from .unified import UnifiedKnowledgeBase
from .vector_store import LocalVectorStore

__all__ = ["UnifiedKnowledgeBase", "LocalVectorStore"]
