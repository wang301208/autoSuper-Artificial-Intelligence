from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, List

from algorithms.base import Algorithm


@dataclass
class Node:
    value: Any
    next: Optional[Node] = None


class LinkedList(Algorithm):
    """A singly linked list implementation.

    Time Complexity:
        - append: O(n)
        - find: O(n)
        - delete: O(n)
    Space Complexity:
        - O(n) where n is the number of stored elements.
    """

    def __init__(self) -> None:
        self.head: Optional[Node] = None

    def append(self, value: Any) -> None:
        """Append a value to the end of the list."""
        new_node = Node(value)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def find(self, value: Any) -> Optional[Node]:
        """Find the first node containing the given value."""
        current = self.head
        while current:
            if current.value == value:
                return current
            current = current.next
        return None

    def delete(self, value: Any) -> bool:
        """Delete the first node containing the value. Returns True if deleted."""
        current = self.head
        prev: Optional[Node] = None
        while current:
            if current.value == value:
                if prev:
                    prev.next = current.next
                else:
                    self.head = current.next
                return True
            prev = current
            current = current.next
        return False

    def to_list(self) -> List[Any]:
        """Return list representation of linked list."""
        result: List[Any] = []
        current = self.head
        while current:
            result.append(current.value)
            current = current.next
        return result

    def execute(self, *args, **kwargs) -> List[Any]:
        """Return list representation of linked list."""
        return self.to_list()
