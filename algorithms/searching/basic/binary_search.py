"""Binary search algorithm."""
from ...base import Algorithm
from typing import Any, List


class BinarySearch(Algorithm):
    """Search a sorted list using the binary search technique."""

    def execute(self, data: List[Any], target: Any) -> int:
        """Return index of ``target`` or -1 if not found.

        Time complexity: O(log n)
        Space complexity: O(1)
        """
        low, high = 0, len(data) - 1
        while low <= high:
            mid = (low + high) // 2
            if data[mid] == target:
                return mid
            if data[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
        return -1
