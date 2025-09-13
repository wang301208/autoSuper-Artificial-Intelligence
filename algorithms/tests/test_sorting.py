import pytest

from algorithms.sorting.basic.bubble_sort import BubbleSort
from algorithms.sorting.basic.quick_sort import QuickSort
from algorithms.sorting.advanced.merge_sort import MergeSort


@pytest.mark.parametrize("algorithm", [BubbleSort(), QuickSort(), MergeSort()])
def test_sorting_basic(algorithm):
    data = [5, 1, 4, 2, 8]
    original = list(data)
    assert algorithm.execute(data) == sorted(data)
    assert data == original


@pytest.mark.parametrize("algorithm", [BubbleSort(), QuickSort(), MergeSort()])
def test_sorting_empty_and_single(algorithm):
    assert algorithm.execute([]) == []
    assert algorithm.execute([1]) == [1]


@pytest.mark.parametrize("algorithm", [BubbleSort(), QuickSort(), MergeSort()])
def test_sorting_type_error(algorithm):
    with pytest.raises(TypeError):
        algorithm.execute([1, "a"])
