"""Linear search algorithm."""
from ...base import Algorithm
from typing import Any, List


class LinearSearch(Algorithm):
    """Find the index of a target value by scanning sequentially."""

    def execute(self, data: List[Any], target: Any) -> int:
        """Return index of ``target`` or -1 if not found.

        Time complexity: O(n)
        Space complexity: O(1)
        """
        for i, value in enumerate(data):
            if value == target:
                return i
        return -1
