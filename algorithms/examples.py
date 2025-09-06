"""Run minimal examples for algorithms."""

from algorithms.sorting.basic.bubble_sort import BubbleSort
from algorithms.sorting.basic.quick_sort import QuickSort
from algorithms.searching.basic.linear_search import LinearSearch
from algorithms.searching.basic.binary_search import BinarySearch
from algorithms.dynamic_programming.basic.fibonacci import Fibonacci
from algorithms.dynamic_programming.basic.lcs import LongestCommonSubsequence
from algorithms.graph.basic.bfs import BreadthFirstSearch
from algorithms.graph.basic.dfs import DepthFirstSearch
from algorithms.storage.basic.lru_cache import LRUCache
from algorithms.storage.basic.lfu_cache import LFUCache
from algorithms.storage.advanced.btree_index import BTreeIndex
from algorithms.utils import Graph


def run_examples() -> None:
    print("BubbleSort:", BubbleSort().execute([3, 1, 2]))
    print("QuickSort:", QuickSort().execute([3, 1, 2]))
    print("LinearSearch:", LinearSearch().execute([1, 2, 3, 4], 3))
    print("BinarySearch:", BinarySearch().execute([1, 2, 3, 4], 3))
    print("Fibonacci:", Fibonacci().execute(7))
    print("LCS:", LongestCommonSubsequence().execute("abcde", "ace"))
    graph = Graph()
    graph.add_edge("A", "B")
    graph.add_edge("A", "C")
    graph.add_edge("B", "D")
    print("BFS:", BreadthFirstSearch().execute(graph, "A"))
    print("DFS:", DepthFirstSearch().execute(graph, "A"))
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")
    cache.put("c", 3)
    print("LRUCache:", cache.execute())
    fcache = LFUCache(2)
    fcache.put("x", 1)
    fcache.put("y", 2)
    fcache.get("x")
    fcache.put("z", 3)
    print("LFUCache:", fcache.execute())
    index = BTreeIndex()
    index.insert(10, "foo")
    index.insert(20, "bar")
    print("BTreeIndex search:", index.search(20))


if __name__ == "__main__":
    run_examples()
