from collections import deque
from typing import Any, Deque, List

from algorithms.base import Algorithm


class Queue(Algorithm):
    """A simple FIFO queue based on ``collections.deque``.

    Time Complexity:
        - enqueue: O(1)
        - dequeue: O(1)
        - peek: O(1)
    Space Complexity:
        - O(n) where n is the number of stored elements.
    """

    def __init__(self) -> None:
        self.items: Deque[Any] = deque()

    def enqueue(self, item: Any) -> None:
        """Add an item to the end of the queue."""
        self.items.append(item)

    def dequeue(self) -> Any:
        """Remove and return the item from the front of the queue."""
        return self.items.popleft() if self.items else None

    def peek(self) -> Any:
        """Return the front item without removing it."""
        return self.items[0] if self.items else None

    def is_empty(self) -> bool:
        """Check whether the queue is empty."""
        return not self.items

    def execute(self, *args, **kwargs) -> List[Any]:
        """Return a snapshot of the current queue."""
        return list(self.items)
