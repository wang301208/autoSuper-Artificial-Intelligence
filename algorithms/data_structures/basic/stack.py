from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from algorithms.base import Algorithm


@dataclass
class Stack(Algorithm):
    """A simple LIFO stack implementation.

    Time Complexity:
        - push: O(1)
        - pop: O(1)
        - peek: O(1)
    Space Complexity:
        - O(n) where n is the number of stored elements.
    """

    items: List[Any]

    def __init__(self) -> None:
        self.items = []

    def push(self, item: Any) -> None:
        """Add an item to the top of the stack."""
        self.items.append(item)

    def pop(self) -> Any:
        """Remove and return the top item of the stack."""
        return self.items.pop() if self.items else None

    def peek(self) -> Any:
        """Return the top item without removing it."""
        return self.items[-1] if self.items else None

    def is_empty(self) -> bool:
        """Check whether the stack is empty."""
        return not self.items

    def execute(self, *args, **kwargs) -> List[Any]:
        """Return a snapshot of the current stack."""
        return list(self.items)
