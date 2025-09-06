"""Quick sort algorithm."""
from ...utils import swap
from ...base import Algorithm
from typing import List, Any


class QuickSort(Algorithm):
    """Sort a list using quick sort."""

    def execute(self, data: List[Any]) -> List[Any]:
        """Return a sorted copy of ``data``.

        Average time complexity: O(n log n)
        Worst-case time complexity: O(n^2)
        Space complexity: O(log n) due to recursion
        """
        arr = data.copy()
        self._quicksort(arr, 0, len(arr) - 1)
        return arr

    def _quicksort(self, arr: List[Any], low: int, high: int) -> None:
        if low < high:
            pivot = self._partition(arr, low, high)
            self._quicksort(arr, low, pivot - 1)
            self._quicksort(arr, pivot + 1, high)

    def _partition(self, arr: List[Any], low: int, high: int) -> int:
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                swap(arr, i, j)
        swap(arr, i + 1, high)
        return i + 1
