"""Bubble sort algorithm."""
from ...utils import swap
from ...base import Algorithm
from typing import List, Any


class BubbleSort(Algorithm):
    """Sort a list using bubble sort."""

    def execute(self, data: List[Any]) -> List[Any]:
        """Return a sorted copy of ``data``.

        Time complexity: O(n^2)
        Space complexity: O(1)
        """
        arr = data.copy()
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    swap(arr, j, j + 1)
        return arr
