import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithms.sorting.basic.bubble_sort import BubbleSort
from algorithms.sorting.basic.quick_sort import QuickSort
from algorithms.searching.basic.linear_search import LinearSearch
from algorithms.searching.basic.binary_search import BinarySearch
from algorithms.graph.basic.bfs import BreadthFirstSearch
from algorithms.graph.basic.dfs import DepthFirstSearch
from algorithms.dynamic_programming.basic.fibonacci import Fibonacci
from algorithms.dynamic_programming.basic.lcs import LongestCommonSubsequence
from algorithms.utils import Graph


def test_bubble_sort():
    data = [5, 2, 9, 1]
    assert BubbleSort().execute(data) == sorted(data)
    assert BubbleSort().execute([]) == []
    assert BubbleSort().execute([3, 3, 1]) == [1, 3, 3]


def test_quick_sort():
    data = [5, 2, 9, 1]
    assert QuickSort().execute(data) == sorted(data)
    assert QuickSort().execute([]) == []
    assert QuickSort().execute([3, 3, 1]) == [1, 3, 3]


def test_linear_search():
    data = [1, 3, 5, 7, 9]
    target = 7
    assert LinearSearch().execute(data, target) == 3
    assert LinearSearch().execute(data, 2) == -1
    assert LinearSearch().execute([], 1) == -1


def test_binary_search():
    data = [1, 3, 5, 7, 9]
    target = 7
    assert BinarySearch().execute(data, target) == 3
    assert BinarySearch().execute(data, 2) == -1
    assert BinarySearch().execute([], 1) == -1
    assert BinarySearch().execute(data, 1) == 0
    assert BinarySearch().execute(data, 9) == 4


def test_breadth_first_search():
    graph = Graph()
    edges = [(1, 2), (1, 3), (2, 4), (3, 4)]
    for u, v in edges:
        graph.add_edge(u, v)
    result = BreadthFirstSearch().execute(graph, 1)
    assert result[0] == 1 and set(result) == {1, 2, 3, 4}
    single = Graph()
    assert BreadthFirstSearch().execute(single, 1) == [1]


def test_depth_first_search():
    graph = Graph()
    edges = [(1, 2), (1, 3), (2, 4), (3, 4)]
    for u, v in edges:
        graph.add_edge(u, v)
    result = DepthFirstSearch().execute(graph, 1)
    assert result[0] == 1 and set(result) == {1, 2, 3, 4}
    single = Graph()
    assert DepthFirstSearch().execute(single, 1) == [1]


def test_fibonacci():
    assert Fibonacci().execute(10) == 55
    assert Fibonacci().execute(0) == 0
    assert Fibonacci().execute(1) == 1


def test_longest_common_subsequence():
    assert LongestCommonSubsequence().execute("abcde", "ace") == 3
    assert LongestCommonSubsequence().execute("", "abc") == 0
    assert LongestCommonSubsequence().execute("abc", "") == 0
