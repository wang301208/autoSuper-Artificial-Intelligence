import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from algorithms.sorting.advanced.heap_sort import HeapSort


def test_heap_sort_basic():
    data = [5, 1, 4, 2, 8]
    original = list(data)
    assert HeapSort().execute(data) == sorted(data)
    assert data == original


def test_heap_sort_descending():
    data = [5, 1, 4, 2, 8]
    assert HeapSort(reverse=True).execute(data) == sorted(data, reverse=True)


def test_heap_sort_empty_and_single():
    sorter = HeapSort()
    assert sorter.execute([]) == []
    assert sorter.execute([1]) == [1]
