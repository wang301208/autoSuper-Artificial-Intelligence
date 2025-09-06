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


def test_sorting_algorithms():
    data = [5, 2, 9, 1]
    assert BubbleSort().execute(data) == sorted(data)
    assert QuickSort().execute(data) == sorted(data)


def test_searching_algorithms():
    data = [1, 3, 5, 7, 9]
    target = 7
    assert LinearSearch().execute(data, target) == 3
    assert BinarySearch().execute(data, target) == 3


def test_graph_algorithms():
    graph = Graph()
    edges = [(1, 2), (1, 3), (2, 4), (3, 4)]
    for u, v in edges:
        graph.add_edge(u, v)
    bfs_result = BreadthFirstSearch().execute(graph, 1)
    dfs_result = DepthFirstSearch().execute(graph, 1)
    assert bfs_result[0] == 1 and set(bfs_result) == {1, 2, 3, 4}
    assert dfs_result[0] == 1 and set(dfs_result) == {1, 2, 3, 4}


def test_dynamic_programming_algorithms():
    assert Fibonacci().execute(10) == 55
    assert LongestCommonSubsequence().execute("abcde", "ace") == 3
