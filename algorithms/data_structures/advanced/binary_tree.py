from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from algorithms.base import Algorithm


@dataclass
class Node:
    value: Any
    left: Optional[Node] = None
    right: Optional[Node] = None


class BinaryTree(Algorithm):
    """A simple binary search tree.

    Time Complexity:
        - insert: O(h)
        - search: O(h)
        - inorder_traversal: O(n)
    Space Complexity:
        - O(n) for storing ``n`` nodes.
    """

    def __init__(self) -> None:
        self.root: Optional[Node] = None

    def insert(self, value: Any) -> None:
        """Insert value into the BST."""
        if self.root is None:
            self.root = Node(value)
            return
        self._insert(self.root, value)

    def _insert(self, node: Node, value: Any) -> None:
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert(node.left, value)
        else:
            if node.right is None:
                node.right = Node(value)
            else:
                self._insert(node.right, value)

    def search(self, value: Any) -> bool:
        """Return True if value is in the tree."""
        return self._search(self.root, value)

    def _search(self, node: Optional[Node], value: Any) -> bool:
        if node is None:
            return False
        if value == node.value:
            return True
        if value < node.value:
            return self._search(node.left, value)
        return self._search(node.right, value)

    def inorder_traversal(self) -> List[Any]:
        """Return inorder list of values."""
        result: List[Any] = []
        self._inorder(self.root, result)
        return result

    def _inorder(self, node: Optional[Node], result: List[Any]) -> None:
        if node:
            self._inorder(node.left, result)
            result.append(node.value)
            self._inorder(node.right, result)

    def execute(self, *args, **kwargs) -> List[Any]:
        """Return inorder traversal of the tree."""
        return self.inorder_traversal()
